#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <cmath>
#include <vector>
#include <tuple>

#include "Eigen/Sparse"

namespace v3_0
{
    class Point
    {
    protected:
        double m_x{}, m_y{};
    public:
        Point() = default;
        Point(double x, double y) : m_x(x), m_y(y) {}
        double x() const { return m_x; }
        double y() const { return m_y; }
    };

    class UnitVector : public Point
    {
    public:
        UnitVector() = default;
        UnitVector(double x, double y)
            : Point(x, y)
            {
                double norm{ std::sqrt(x * x + y * y) };
                m_x /= norm;
                m_y /= norm;
            }
    };

    class Ray
    {
        Point m_origin;
        UnitVector m_direction;
    public:
        Ray() = default;
        Ray(Point origin, UnitVector direction)
            : m_origin(origin), m_direction(direction)
            {
            }
        const Point& getOrigin() const { return m_origin; }
        const UnitVector& getDirection() const { return m_direction; }
    };

    /****************************
     * CLASS PARALLELPROJECTION *
     ****************************/

    class ParallelProjection
    {
        using WeightAndPixelType = std::tuple<double, Point>;

    public:
        enum IntersectionType
        {
            DIRAC,      // Closest pixel
            LENGTH,     // Intersection length
            AREA        // Intersection area
        };

    protected:
        int m_nProj{}; // Number of projections
        int m_nRays{}; // Number of rays per projection
        double m_angleRange{}; // Angular range in radians
        int m_imageSize{};     // Image size in pixels
        IntersectionType m_intersectionType{ IntersectionType::DIRAC };

        // First index = #projection
        // Second index = #ray in the projection
        std::vector<std::vector<Ray>> m_rays;
        void computeRays();
        std::vector<WeightAndPixelType> computeWeightsAndPixels(int p, int r);

        template<typename T>
        using MatrixType = Eigen::SparseMatrix<T, Eigen::RowMajor>;
        template<typename T>
        MatrixType<T> computeSystemMatrix();

    public:
        ParallelProjection(int proj, int ray, double range, int size)
            : m_nProj(proj), m_nRays(ray), m_angleRange(range), m_imageSize(size)
            {
            }
    };

    void ParallelProjection::computeRays()
    {
        m_rays.resize(m_nProj);

        double angularStep{ m_angleRange / m_nProj };
        double rayStep{ static_cast<double>(m_imageSize) / m_nRays };
        double angle{};

        for (int p{}; p < m_nProj; ++p, angle += angularStep)
        {
            m_rays[p].resize(m_nRays);

            double cosAngle{ std::cos(angle) };
            double sinAngle{ std::sin(angle) };
            UnitVector direction( -sinAngle, cosAngle );

            for (int r{}; r < m_nRays; ++r)
            {
                double x{ (r - m_nRays / 2) * rayStep * cosAngle };
                double y{ (r - m_nRays / 2) * rayStep * sinAngle };
                Point origin(x, y);
                m_rays[p][r] = Ray(origin, direction);
            }
        }
    }

    std::vector<ParallelProjection::WeightAndPixelType> ParallelProjection::computeWeightsAndPixels(int p, int r)
    {
        std::vector<ParallelProjection::WeightAndPixelType> weightsAndPixels;

        auto ray{ m_rays[p][r] };

        auto diagonal{ static_cast<int>(std::sqrt(2.0) * m_imageSize) };

        switch (m_intersectionType)
        {
        case DIRAC:
            for (int lambda{ -diagonal / 2}; lambda <= diagonal / 2; ++lambda)
            {

                auto i{ ray.getOrigin().x() + lambda * ray.getDirection().x() + m_imageSize / 2 };
                auto j{ ray.getOrigin().y() + lambda * ray.getDirection().y() + m_imageSize / 2 };
                if (i >= 0 && i < m_imageSize && j >=0 && j < m_imageSize)
                {
                    Point pixel(i, j);
                    weightsAndPixels.push_back(std::make_tuple(1.0, pixel));
                }
            }
            break;

        case LENGTH:
            /* code */
            break;

        case AREA:
            /* code */
            break;

        default:
            break;
        }

        return weightsAndPixels;
    }

    template<typename T>
    ParallelProjection::MatrixType<T> ParallelProjection::computeSystemMatrix()
    {
        MatrixType<T> A( m_nProj * m_nRays, m_imageSize * m_imageSize);
        // Reserve memory for non null coefficients
        // Maximum number of intersections per ray equals sqrt(2) * image size
        const int maxIntersections{ static_cast<int>(std::sqrt(2.0) * m_imageSize) };
        A.reserve( Eigen::VectorXi::Constant(m_nProj * m_nRays, maxIntersections) );

        computeRays();
        for (int p{}; p < m_nProj; ++p)
            for (int r{}; r < m_nRays; ++r)
            {
                auto weightsAndPixels{ computeWeightsAndPixels(p, r) };
                for (const auto weightAndPixel : weightsAndPixels)
                {
                    auto [weight, pixel] { weightAndPixel };
                    auto i{ static_cast<int>(pixel.x()) };
                    auto j{ static_cast<int>(pixel.y()) };
                    A.coeffRef(p * m_nRays + r, j * m_imageSize + i) += static_cast<T>(weight);
                }
            }
        A.makeCompressed(); // Remove extra coefficients
        return A;
    }

    /******************
     * CLASS SINOGRAM *
     ******************/
    template <typename TInputPixel, typename TOutputPixel = TInputPixel>
    class Sinogram : public Processing1<TInputPixel, TOutputPixel>, ParallelProjection
    {
    public:
        Sinogram(Image<TInputPixel>&inputImage, int nProj, int nRays, double angleRange, int imageSize, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace, nRays, nProj),
              ParallelProjection(nProj, nRays, angleRange, imageSize) {}
        void setIntersectionType(IntersectionType type) { m_intersectionType = type; }
        void process() override;
    };

    template <typename TInputPixel, typename TOutputPixel>
    void Sinogram<TInputPixel, TOutputPixel>::process()
    {
        computeRays();

        for (int r{}; r < m_nRays; ++r)
            for (int p{}; p < m_nProj; ++p)
            {
                auto weightsAndPixels{ computeWeightsAndPixels(p, r) };
                double sum{};
                for (const auto& weightAndPixel : weightsAndPixels)
                {
                    // Retrieve pixel location and weight
                    auto [weight, pixel] { weightAndPixel };
                    // Closest pixel with integer coordinates
                    auto i{ static_cast<int>(pixel.x()) };
                    auto j{ static_cast<int>(pixel.y()) };
                    sum += weight * this->m_inputImage(i, j);
                }
                this->m_outputImage(r, p) = static_cast<TOutputPixel>(sum);
            }
    }

    /************************
     * CLASS RECONSTRUCTION *
     ************************/
    template<typename TInputPixel, typename TOutputPixel = TInputPixel>
    class Reconstruction : public Processing1<TInputPixel, TOutputPixel>,
                           public ParallelProjection
    {
    public:
        enum ReconstructionType
        {
            FBP,    // Filtered BackProjection
            ART,    // Algebraic Reconstruction Method
            MLEM    // Maximum Likelihood Expectation Maximization
        };

        enum FilterType
        {
            NONE,       // No filtering
            RAMP,       // Ramp high-pass filtering (theoretical)
            BUTTERWORTH // Ramp and Butterworth low-pass filtering
        };

    private:
        ReconstructionType m_reconstructionType{ FBP };
        FilterType m_filterType{ NONE };
        double m_cutoffFrequency{ 0.5 };    // Normalized cutoff frequency (max 0.5)

        template<typename T>
        using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>; 
        template<typename T>
        VectorType<T> convertImage2Vector(const Image<T>& image);
        template<typename T>
        void convertVector2Image(const VectorType<T>& vector, Image<T>& image);

    public:
        Reconstruction(Image<TInputPixel>& inputImage, double angleRange, int imageSize, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace, imageSize, imageSize),
              ParallelProjection(inputImage.getHeight(), inputImage.getWidth(),
                                 angleRange, imageSize) {}
        void setReconstructionType(ReconstructionType type) { m_reconstructionType = type; }
        void setFilterType(FilterType type) { m_filterType = type; }
        void setCutoffFrequency(double f) { m_cutoffFrequency = f; }
        void filteredBackProjection();
        void process() override;
    };

    template<typename TInputPixel, typename TOutputPixel>
    void Reconstruction<TInputPixel, TOutputPixel>::filteredBackProjection()
    {
        // Projection prefiltering
        auto filteredSinogram { this->m_inputImage };
        if (m_filterType != NONE)
        {
            // Build transfer function
            auto filter { [=](int i, int j)
                {
                    double high{ std::abs(2.0 * i / m_nRays) }, low{ };
                    switch (m_filterType)
                    {
                    case RAMP:
                        low = 1.0;
                        break;
                    case BUTTERWORTH:
                        {
                            constexpr int n{ 2 };
                            double fx{ i / m_cutoffFrequency / m_nRays }, fx2{ fx * fx }, tmp{ fx2 };
                            for (int k{}; k < n - 1; ++k) tmp *= fx2;
                            low = 1.0 / (1.0 + (std::sqrt(2) - 1) * tmp);
                        }
                        break;
                    }
                    return high * low;
                }
            };
            Image<TInputPixel> mtf(m_nRays, m_nProj);
            mtf.fill(filter);
            //auto out { mtf.convert<unsigned char>(true) };
            //out.write("mtf.raw");
            filteredSinogram = v3_0::frequentialFiltering<TInputPixel>(this->m_inputImage, mtf, false);
        }

        // Compute rays (origin + direction)
        computeRays();

        // Back projection
        for (int r{}; r < m_nRays; ++r)
            for (int p{}; p < m_nProj; ++p)
            {
                // Projection value
                auto proj{ filteredSinogram(r, p) };

                if (proj != 0.0)
                {
                    auto weightsAndPixels{ computeWeightsAndPixels(p, r) };
                    for (const auto& weightAndPixel : weightsAndPixels)
                    {
                        auto [weight, pixel] { weightAndPixel };
                        auto i{ static_cast<int>(pixel.x()) };
                        auto j{ static_cast<int>(pixel.y()) };
                        this->m_outputImage(i, j) += static_cast<TOutputPixel>(weight * proj);
                    }
                }
            }
    }

    template<typename TInputPixel, typename TOutputPixel>
    template<typename T>
    Reconstruction<TInputPixel, TOutputPixel>::VectorType<T> Reconstruction<TInputPixel, TOutputPixel>::convertImage2Vector(const Image<T>& image)
    {
        Reconstruction<TInputPixel, TOutputPixel>::VectorType<T> vector(image.getWidth() * image.getHeight());
        for (int j{}, k{}; j < image.getHeight(); ++j)
            for (int i{}; i < image.getWidth(); ++i)
                vector(k++) = image(i, j);
        return vector;
    }

    template<typename TInputPixel, typename TOutputPixel>
    template<typename T>
    void Reconstruction<TInputPixel, TOutputPixel>::convertVector2Image(const Reconstruction<TInputPixel, TOutputPixel>::VectorType<T>& vector,
        Image<T>& image)
    {
        for (int j{}, k{}; j < image.getHeight(); ++j)
            for (int i{}; i < image.getWidth(); ++i)
                image(i, j) = vector(k++);
    }

    template<typename TInputPixel, typename TOutputPixel>
    void Reconstruction<TInputPixel, TOutputPixel>::process()
    {
        // Filtered BackProjection
        if (m_reconstructionType == FBP)
        {
            std::cout << "Filtered BackProjection\n";
            filteredBackProjection();
            return;
        }

        // Build algebraic system to solve
        auto A{ computeSystemMatrix<TOutputPixel>() };
        auto b{ convertImage2Vector<TOutputPixel>(this->m_inputImage) };

        constexpr int nIter{ 100 };

        switch (m_reconstructionType)
        {
        case ART:
            {
                std::cout << "Algebraic Reconstruction Technique\n";

                // Set reconstructed pixel vector to 0
                VectorType<TOutputPixel> f{ VectorType<TOutputPixel>::Zero(A.cols()) };

                // Precompute inner product of A rows
                VectorType<TOutputPixel> innerProduct(A.rows());
                for (int i{}; i < A.rows(); ++i)
                    innerProduct(i) = A.row(i).dot(A.row(i));

                for (int n{}; n < nIter; ++n)
                {
                    std::cout << "\tIteration #" << n << '\n';
                    for (int i{}; i < A.rows(); ++i)
                    {
                        auto coef{ (b(i) - A.row(i).dot(f)) / innerProduct(i) };
                        f += coef * A.row(i).transpose();
                    }
                }

                // Positivity constraint
                for (int i{}; i < A.cols(); ++i)
                    if (f(i) < 0.0)
                        f(i) = 0.0;

                convertVector2Image(f, this->m_outputImage);
            }
            break;
        
        case MLEM:
            {
                std::cout << "Maximum Likelihood Expectation Maximization\n";

                // Set reconstructed pixel vector to 0
                VectorType<TOutputPixel> f{ VectorType<TOutputPixel>::Ones(A.cols()) };

                // Precompute normalization factor
                VectorType<TOutputPixel> normFactor{ VectorType<TOutputPixel>::Zero(A.cols()) };
                for (int i{}; i < A.rows(); ++i)
                    normFactor += A.row(i);

                for (int n{}; n < nIter; ++n)
                {
                    std::cout << "\tIteration #" << n << '\n';

                    VectorType<TOutputPixel> coef{ VectorType<TOutputPixel>::Zero(A.cols()) };
                    for (int i{}; i < A.rows(); ++i)
                    {
                        auto den{ A.row(i) * f };
                        if (den > 0.0)
                            coef += A.row(i) * b(i) / den;
                    }

                    for (int j{}; j < A.cols(); ++j)
                        if (normFactor(j) != 0.0)
                            f(j) *= coef(j) / normFactor(j);
                }

                convertVector2Image(f, this->m_outputImage);
            }
            break;
        }


    }
}

#endif // RECONSTRUCTION_H
