#ifndef PROCESSING_H
#define PROCESSING_H

#include <type_traits>
#include <random>
#include <map>
#include <ctime>

#include "image.h"
#include "FFT.h"
#include "SFINAE.h"

namespace v2_0
{
    /****************
     * BASE CLASSES *
     ****************/

    // Base class for processing between two images
    // inPlace processing is only possible if TOutputPixel can be converted to TInputPixel1
    // generic template class to manage incompatible image types
    template <typename TInputPixel1, typename TInputPixel2, typename TOutputPixel, typename sfinae = void>
    class Processing2
    {
    protected:
        Image<TInputPixel1>& m_inputImage1;
        const Image<TInputPixel2>& m_inputImage2;
        Image<TOutputPixel> m_outputImage;

    public:
        Processing2(Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2, bool inPlace = false);
        virtual void process() = 0;
        void update()
        {
            process();
        }
        Image<TOutputPixel> getOutput() { return m_outputImage; }
    };

    template <typename TInputPixel1, typename TInputPixel2, typename TOutputPixel, typename sfinae>
    Processing2<TInputPixel1, TInputPixel2, TOutputPixel, sfinae>::Processing2(Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2, bool inPlace)
        : m_inputImage1(inputImage1), m_inputImage2(inputImage2)
    {
        // Create output image
        m_outputImage = Image<TOutputPixel>(inputImage1.getWidth(), inputImage1.getHeight());

        if (inPlace)
            std::cerr << "Inplace processing not possible because of incompatible input and output types" << '\n';
    }

    // specialized template class to manage compatible image types
    template <typename TInputPixel1, typename TInputPixel2, typename TOutputPixel>
    class Processing2<TInputPixel1, TInputPixel2, TOutputPixel, SameORCastablePolicy<TOutputPixel, TInputPixel1>>
    {
    protected:
        Image<TInputPixel1>& m_inputImage1;
        const Image<TInputPixel2>& m_inputImage2;
        Image<TOutputPixel> m_outputImage;
        bool m_inPlace;
    public:
        Processing2(Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2, bool inPlace = false);
        virtual void process() = 0;
        void update()
        {
            process();
            if (m_inPlace)
            {
                for (int i{ 0 }; i < m_inputImage1.getWidth(); ++i)
                    for (int j{ 0 }; j < m_inputImage1.getHeight(); ++j)
                        m_inputImage1(i, j) = static_cast<TInputPixel1>(m_outputImage(i, j));
            }
        }
        Image<TOutputPixel> getOutput() { return m_outputImage; }
    };

    template <typename TInputPixel1, typename TInputPixel2, typename TOutputPixel>
    Processing2<TInputPixel1, TInputPixel2, TOutputPixel, SameORCastablePolicy<TOutputPixel, TInputPixel1>>
        ::Processing2(Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2, bool inPlace)
            : m_inputImage1(inputImage1), m_inputImage2(inputImage2), m_inPlace(inPlace)
    {
        // Create output image
        m_outputImage = Image<TOutputPixel>(inputImage1.getWidth(), inputImage1.getHeight());
    }

    // Base class for processing of one image
    // inPlace processing is only possible if TOutputPixel can be converted to TInputPixel
    // generic template class to manage incompatible image types
    template <typename TInputPixel, typename TOutputPixel, typename sfinae = void>
    class Processing1
    {
    protected:
        Image<TInputPixel>& m_inputImage;
        Image<TOutputPixel> m_outputImage;
    public:
        Processing1(Image<TInputPixel>& inputImage, bool inPlace = false, int widthOut = -1, int heightOut = -1);
        virtual void process() = 0;
        void update()
        {
            process();
        }
        Image<TOutputPixel> getOutput() { return m_outputImage; }
    };

    template <typename TInputPixel, typename TOutputPixel, typename sfinae>
    Processing1<TInputPixel, TOutputPixel, sfinae>::Processing1(Image<TInputPixel>& inputImage, bool inPlace, int widthOut, int heightOut)
        : m_inputImage(inputImage)
    {
        // Create output image
        if ( widthOut == -1 && heightOut == -1 )
            m_outputImage = Image<TOutputPixel>(inputImage.getWidth(), inputImage.getHeight());
        else
            m_outputImage = Image<TOutputPixel>(widthOut, heightOut);

        if (inPlace)
            std::cerr << "Inplace processing not possible because of incompatible input and output types" << '\n';
    }

    // specialized template class to manage compatible image types
    template <typename TInputPixel, typename TOutputPixel>
    class Processing1<TInputPixel, TOutputPixel, SameORCastablePolicy<TOutputPixel, TInputPixel>>
    {
    protected:
        Image<TInputPixel>& m_inputImage;
        Image<TOutputPixel> m_outputImage;
        bool m_inPlace;
    public:
        Processing1(Image<TInputPixel>& inputImage, bool inPlace = false, int widthOut = -1, int heightOut = -1);
        virtual void process() = 0;
        void update()
        {
            process();
            if (m_inPlace)
            {
                for (int i{ 0 }; i < m_inputImage.getWidth(); ++i)
                    for (int j{ 0 }; j < m_inputImage.getHeight(); ++j)
                        m_inputImage(i, j) = static_cast<TInputPixel>(m_outputImage(i, j));
            }
        }
        Image<TOutputPixel> getOutput() { return m_outputImage; }
    };

    template <typename TInputPixel, typename TOutputPixel>
    Processing1<TInputPixel, TOutputPixel, SameORCastablePolicy<TOutputPixel, TInputPixel>>
        ::Processing1(Image<TInputPixel>& inputImage, bool inPlace, int widthOut, int heightOut)
            : m_inputImage(inputImage), m_inPlace(inPlace)
    {
        // Create output image
        if ( widthOut == -1 && heightOut == -1 )
            m_outputImage = Image<TOutputPixel>(inputImage.getWidth(), inputImage.getHeight());
        else
            m_outputImage = Image<TOutputPixel>(widthOut, heightOut);
    }

    /***********************
     * PROCESSING EXAMPLES *
     ***********************/

    // Addition between two images (template function version)
    template <typename TInputPixel1, typename TInputPixel2 = TInputPixel1, typename TOutputPixel = TInputPixel1>
    Image<TOutputPixel> addition(const Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2)
    {
        Image<TOutputPixel> outputImage(inputImage1.getWidth(), inputImage1.getHeight());
        for (int i{ 0 }; i < outputImage.getWidth(); ++i)
            for (int j{ 0 }; j < outputImage.getHeight(); ++j)
                outputImage(i, j) = inputImage1(i, j) + inputImage2(i, j);
        return outputImage;
    }

    // Addition between two images (template class version)
    template <typename TInputPixel1, typename TInputPixel2 = TInputPixel1, typename TOutputPixel = TInputPixel1>
    class Addition : public Processing2<TInputPixel1, TInputPixel2, TOutputPixel>
    {
    public:
        Addition(Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2, bool inPlace = false)
            : Processing2<TInputPixel1, TInputPixel2, TOutputPixel>(inputImage1, inputImage2, inPlace) {}
        void process();
    };

    template <typename TInputPixel1, typename TInputPixel2, typename TOutputPixel>
    void Addition<TInputPixel1, TInputPixel2, TOutputPixel>::process()
    {
        for (int i{ 0 }; i < this->m_outputImage.getWidth(); ++i)
            for (int j{ 0 }; j < this->m_outputImage.getHeight(); ++j)
                this->m_outputImage(i, j) = this->m_inputImage1(i, j) + this->m_inputImage2(i, j);
    }

    // Addition between one image and a scalar (template function version)
    template <typename TInputPixel, typename TOutputPixel = TInputPixel>
    Image<TOutputPixel> additionScalar(const Image<TInputPixel>& inputImage, TInputPixel scalar)
    {
        Image<TOutputPixel> outputImage(inputImage.getWidth(), inputImage.getHeight());
        for (int i{ 0 }; i < outputImage.getWidth(); ++i)
            for (int j{ 0 }; j < outputImage.getHeight(); ++j)
                outputImage(i, j) = inputImage(i, j) + scalar;
        return outputImage;
    }

    // Addition between one image and a scalar (template class version)
    template <typename TInputPixel, typename TOutputPixel = TInputPixel>
    class AdditionScalar : public Processing1<TInputPixel, TOutputPixel>
    {
        TInputPixel m_scalar;

    public:
        AdditionScalar(Image<TInputPixel>& inputImage, TInputPixel scalar, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace), m_scalar(scalar) {}
        void process();
    };

    template <typename TInputPixel, typename TOutputPixel>
    void AdditionScalar<TInputPixel, TOutputPixel>::process()
    {
        for (int i{ 0 }; i < this->m_outputImage.getWidth(); ++i)
            for (int j{ 0 }; j < this->m_outputImage.getHeight(); ++j)
                this->m_outputImage(i, j) = this->m_inputImage(i, j) + this->m_scalar;
    }

    /**************************
     * HISTOGRAM EQUALIZATION *
     **************************/

    template<typename TInputPixel, typename TOutputPixel = TInputPixel>
    class Equalization : public Processing1<TInputPixel, TOutputPixel>
    {
        using histogram_type = std::vector<double>;
        histogram_type m_histogram, m_cumulated;
        TInputPixel m_minOut, m_maxOut;
        void computeHistogram();

    public:
        Equalization(Image<TInputPixel>& inputImage, TInputPixel minOut, TInputPixel maxOut, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace),
                m_minOut(minOut), m_maxOut(maxOut) {}
        void process() override;
    };

    template<typename TInputPixel, typename TOutputPixel>
    void Equalization<TInputPixel, TOutputPixel>::computeHistogram()
    {
        const auto histogramSize{ static_cast<std::size_t>(std::numeric_limits<TInputPixel>::max() + 1) };

        // Compute histogram
        m_histogram.resize(histogramSize);
        for (int i{}; i < this->m_inputImage.getWidth(); ++i)
            for (int j{}; j < this->m_inputImage.getHeight(); ++j)
            {
                auto pixelValue{ static_cast<std::size_t>(this->m_inputImage(i,j)) };
                m_histogram[pixelValue]++;
            }

        for (auto& prob : m_histogram)
            prob /= this->m_inputImage.getWidth() * this->m_inputImage.getHeight();

        // Compute cumulated histogram
        m_cumulated.resize(histogramSize);
        m_cumulated[0] = m_histogram[0];
        for (std::size_t k{ 1 }; k < m_histogram.size(); ++k)
            m_cumulated[k] = m_cumulated[k - 1] + m_histogram[k];
    }

    template<typename TInputPixel, typename TOutputPixel>
    void Equalization<TInputPixel, TOutputPixel>::process()
    {
        computeHistogram();

        for (int i{}; i < this->m_outputImage.getWidth(); ++i)
            for (int j{}; j < this->m_outputImage.getHeight(); ++j)
            {
                auto inputPixelValue{ this->m_inputImage(i,j) };
                auto histogramIndex{ static_cast<std::size_t>(inputPixelValue) };
                auto outputPixelValue{ (m_maxOut - m_minOut) * m_cumulated[histogramIndex] + m_minOut };
                this->m_outputImage(i, j) = static_cast<TOutputPixel>(outputPixelValue);
            }
    }

    /*************************
     * CONVOLUTION FILTERING *
     *************************/

    template <typename TInputPixel1, typename TInputPixel2 = TInputPixel1, typename TOutputPixel = TInputPixel1>
    Image<TOutputPixel> convolution(const Image<TInputPixel1>& inputImage, const Image<TInputPixel2>& kernel)
    {
        Image<TOutputPixel> outputImage(inputImage.getWidth(), inputImage.getHeight());

        const auto nx{ kernel.getWidth() / 2}, ny{ kernel.getHeight() / 2 };

        for (int i{ nx }; i < outputImage.getWidth() - nx; ++i)
            for (int j{ ny }; j < outputImage.getHeight() - ny; ++j)
            {
                for (int k{ -nx }; k <= nx; ++k)
                    for (int l{ -ny }; l <= ny; ++l)
                        outputImage(i, j) += kernel(k + nx, l + ny) * inputImage(i + k, j + l);
            }

        return outputImage;
    }

    /*************************
     * FREQUENTIAL FILTERING *
     *************************/

    template <typename TInputPixel1, typename TInputPixel2 = TInputPixel1, typename TOutputPixel = TInputPixel1>
    Image<TOutputPixel> frequentialFiltering(const Image<TInputPixel1>& inputImage, const Image<TInputPixel2>& mtf,
                                             bool filter2D = true)
    {
        // Image for null imaginary part
        Image<TInputPixel1> imaginaryIn(inputImage.getWidth(), inputImage.getHeight());

        // Images for input spectrum
        Image<TInputPixel1> realSpectrum(inputImage.getWidth(), inputImage.getHeight());
        Image<TInputPixel1> imaginarySpectrum(inputImage.getWidth(), inputImage.getHeight());

        // Direct Fourier transform
        if (filter2D)
            directFFT(inputImage, imaginaryIn, realSpectrum, imaginarySpectrum);
        else
            directFFT1DX(inputImage, imaginaryIn, realSpectrum, imaginarySpectrum);

        // Multiplication with modulation transfer function
        for (int i{}; i < mtf.getWidth(); ++i)
            for (int j{}; j < mtf.getHeight(); ++j)
            {
                realSpectrum(i, j) *= mtf(i, j);
                imaginarySpectrum(i, j) *= mtf(i, j);
            }

        // Images for output real and imaginary parts
        Image<TOutputPixel> realOut(inputImage.getWidth(), inputImage.getHeight());
        Image<TOutputPixel> imaginaryOut(inputImage.getWidth(), inputImage.getHeight());

        // Inverse Fourier transform
        if (filter2D)
            inverseFFT(realSpectrum, imaginarySpectrum, realOut, imaginaryOut);
        else
            inverseFFT1DX(realSpectrum, imaginarySpectrum, realOut, imaginaryOut);

        return realOut;
    }

    /*****************
     * GRADIENT NORM *
     *****************/

    template <typename TInputPixel1, typename TInputPixel2 = TInputPixel1, typename TOutputPixel = TInputPixel1>
    Image<TOutputPixel> norm(const Image<TInputPixel1>& inputImage1, const Image<TInputPixel2>& inputImage2)
    {
        Image<TOutputPixel> outputImage(inputImage1.getWidth(), inputImage1.getHeight());

        for (int i{}; i < outputImage.getWidth(); ++i)
            for (int j{}; j < outputImage.getHeight(); ++j)
            {
                auto n{ std::sqrt(inputImage1(i, j) * inputImage1(i, j) + inputImage2(i, j) * inputImage2(i, j)) };
                outputImage(i, j) = static_cast<TOutputPixel>(n);
            }

        return outputImage;
    }

    /**********************
     * MULTI THRESHOLDING *
     **********************/

    template<typename TInputPixel, typename TOutputPixel = TInputPixel>
    class MultiThresholding : public Processing1<TInputPixel, TOutputPixel>
    {
        std::vector<TInputPixel> m_thresholds;

    public:
        MultiThresholding(Image<TInputPixel>& inputImage, const std::vector<TInputPixel>& thresholds, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace), m_thresholds(thresholds) {}
        void process() override;
    };

    template<typename TInputPixel, typename TOutputPixel>
    void MultiThresholding<TInputPixel, TOutputPixel>::process()
    {
        for (int i{}; i < this->m_outputImage.getWidth(); ++i)
            for (int j{}; j < this->m_outputImage.getHeight(); ++j)
            {
                auto value{ this->m_inputImage(i, j) };
                auto f{ [value](int threshold) { return threshold > value; } };
                auto it{ std::find_if(m_thresholds.begin(), m_thresholds.end(), f) };
                this->m_outputImage(i, j) = static_cast<TOutputPixel>(it - m_thresholds.begin());
            }
    }

    /*********************
     * OTSU THRESHOLDING *
     *********************/


    /**************************
     * K-MEANS CLASSIFICATION *
     **************************/

    // K-means classification (template class version)

    // Generic non implemented template class
    template <typename TInputPixel, typename TOutputPixel = TInputPixel, typename sfinae = void>
    class KMeans : public Processing1<TInputPixel, TOutputPixel>
    {
    public:
        KMeans(Image<TInputPixel>& inputImage, int nClasses, int nDim, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace)
        {
            std::cerr << "Input pixel type has neither operator[] nor size() member function" << '\n';
        }
        void process(){};
    };

    // Specialization for floating point images
    template <typename TInputPixel, typename TOutputPixel>
    class KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>> : public Processing1<TInputPixel, TOutputPixel>
    {
        int m_nClasses, m_nDim;
        std::vector<std::vector<double>> m_g;
        std::vector<int> m_nPixels;

        void initCenters();
        void affectation();
        void representation();
        double variance();
        double distance(const TInputPixel& pixel, const std::vector<double>& g);

        public:
        KMeans(Image<TInputPixel>& inputImage, int nClasses, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace),
                m_nClasses(nClasses), m_nDim(inputImage(0, 0).size())
            {
                m_nPixels.resize(m_nClasses);
                m_g.resize(m_nClasses);
                for (int k{}; k < m_nClasses; ++k)
                    m_g[k].resize(m_nDim);
            }
            void process() override;
            std::ostream& concordanceMatrix(std::ostream& out, const Image<TOutputPixel>& groundTruth);
    };

    template <typename TInputPixel, typename TOutputPixel>
    void KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>::initCenters()
    {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(0, 255);

        // Background class center
        for (int l{}; l < m_nDim; ++l)
            m_g[0][l] = 0;

        // Other class centers
        for (int k{ 1 }; k < m_nClasses; ++k)
            for (int l{}; l < m_nDim; ++l)
                m_g[k][l] = distrib(gen);
    }

    // Compute distance between a pixel and a given class center
    template <typename TInputPixel, typename TOutputPixel>
    double KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>
        ::distance(const TInputPixel& pixel, const std::vector<double>& g)
    {
        double d2{};
        for (int l{}; l < m_nDim; ++l)
        {
            auto diff{ pixel[l] - g[l] };
            d2 += diff * diff;
        }
        return d2;
    }

    template <typename TInputPixel, typename TOutputPixel>
    void KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>::affectation()
    {
        for (int i{}; i < this->m_inputImage.getWidth(); ++i)
            for (int j{}; j < this->m_inputImage.getHeight(); ++j)
            {
                auto pixel{ this->m_inputImage(i, j) };

                auto dMin{ distance(pixel, m_g[0]) };
                int kMin{ 0 };
                for (int k{ 1 }; k < m_nClasses; ++k)
                {
                    auto d{ distance(pixel, m_g[k]) };
                    if (d < dMin)
                    {
                        dMin = d;
                        kMin = k;
                    }
                }

                this->m_outputImage(i, j) = static_cast<TOutputPixel>(kMin);
            }
    }

    template <typename TInputPixel, typename TOutputPixel>
    void KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>::representation()
    {
        // Initialisation des coordonnées des noyaux à zéro
        for (int k{}; k < m_nClasses; ++k)
        {
            m_nPixels[k] = 0;
            for (int l{}; l < m_nDim; ++l)
                m_g[k][l] = 0.0;
        }

        // Calcul de la somme des coordonnées des pixels appartenant à chaque classe
        for (int i{}; i < this->m_inputImage.getWidth(); ++i)
            for (int j{}; j < this->m_inputImage.getHeight(); ++j)
            {
                auto k{ this->m_outputImage(i, j) }; // Numéro de classe du pixel
                auto pixel{ this->m_inputImage(i, j) }; // Vecteur d'attributs du pixel

                m_nPixels[k]++;
                for (int l{}; l < m_nDim; ++l)
                    m_g[k][l] += pixel[l];
            }

        // Normalisation des sommes au nombre de pixels par classe
        for (int k{}; k < m_nClasses; ++k)
            for (int l{}; l < m_nDim; ++l)
                m_g[k][l] /= m_nPixels[k];

    }

    template <typename TInputPixel, typename TOutputPixel>
    double KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>::variance()
    {
        double var{};
        for (int i{}; i < this->m_inputImage.getWidth(); ++i)
            for (int j{}; j < this->m_inputImage.getHeight(); ++j)
            {
                auto pixel{ this->m_inputImage(i, j) };
                auto k{ this->m_outputImage(i, j) };
                var += distance(pixel, m_g[k]);
            }
        return var;
    }

    template <typename TInputPixel, typename TOutputPixel>
    void KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>::process()
    {
        // Initialisation aléatoire des centres de classes
        initCenters();

        // Affectation des pixels aux classes dont le centre est le plus proche
        affectation();

        // Itérations
        double var0, var{ variance() };
        int it{};
        do {
            // Sauvegarde de la somme des variances intra-classe à l'itération précédente
            var0 = var;

            // Mise à jour des centres de classes
            representation();

            // Affectation des pixels aux classes dont le centre est le plus proche
            affectation();

            // Mise à jour de la somme des variances intra-classe
            var = variance();

            // Affichage
            std::cout << "Iteration " << it << " -> " << std::abs(var - var0) / var0 << '\n';

            ++it;
        }
        while (std::abs(var - var0) / var0 > 1.e-6);
    }

    template <typename TInputPixel, typename TOutputPixel>
    std::ostream& KMeans<TInputPixel, TOutputPixel, BracketsANDSizePolicy<TInputPixel>>
        ::concordanceMatrix(std::ostream& out, const Image<TOutputPixel>& groundTruth)
    {

        return out;
    }


    /*************************
     * GAUSSIAN DISTRIBUTION *
     *************************/

    // Simulation of Gaussian random distributions from labels (template function version)
    template <typename TInputPixel, typename TOutputPixel = TInputPixel>
    Image<std::vector<TOutputPixel>> gaussianDistributions(const Image<TInputPixel>& inputImage, const int nDim, const double stdDev)
    {
        // Input image traversal to list unique label value and associate to nDim vector of Gaussian distribution means
        TOutputPixel maxValue{ static_cast<TOutputPixel>(std::numeric_limits<TOutputPixel>::max()) };
        std::srand(std::time(nullptr));
        std::map<TInputPixel, std::vector<double>> labelMap;
        labelMap[0] = std::vector<double>(nDim, 3 * stdDev);   // Means equal to 0 for background
        for (int i{ 0 }; i < inputImage.getWidth(); ++i)
            for (int j{ 0 }; j < inputImage.getHeight(); ++j)
            {
                TInputPixel value{ inputImage(i, j) };
                if ( labelMap.find(value) == labelMap.end() )
                {
                    std::vector<double> v;
                    for (int k{ 0 }; k < nDim; ++k)
                    {
                        double mean{ static_cast<double>(std::rand() % maxValue) };
                        v.push_back(mean);
                    }
                    labelMap[value] = v;
                }
            }

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{0.0, 1.0};
        Image<std::vector<TOutputPixel>> outputImage(inputImage.getWidth(), inputImage.getHeight());
        for (int i{ 0 }; i < outputImage.getWidth(); ++i)
            for (int j{ 0 }; j < outputImage.getHeight(); ++j)
            {
                TInputPixel value{ inputImage(i, j) };
                std::vector<TOutputPixel> v(nDim);
                for (int k{ 0 }; k < nDim; ++k)
                {
                    double mean{ static_cast<double>(labelMap[value][k]) };
                    double coef{ mean + stdDev * d(gen) };
                    v[k] = ( coef >= 0.0 )
                            ? ((coef < maxValue) ? static_cast<TOutputPixel>(coef) : maxValue )
                            : 0;
                }
                outputImage(i, j) = v;
            }

        return outputImage;
    }

    /************************
     * POISSON DISTRIBUTION *
     ************************/

    // Simulation of Poisson random distributions (template class version)
    template<typename TInputPixel, typename TOutputPixel = TInputPixel>
    class PoissonNoise : public Processing1<TInputPixel, TOutputPixel>
    {
        TInputPixel m_nbPhotonMax;
    
    public:
        PoissonNoise(Image<TInputPixel>& inputImage, TInputPixel nbPhotonMax, bool inPlace = false)
            : Processing1<TInputPixel, TOutputPixel>(inputImage, inPlace),
              m_nbPhotonMax(nbPhotonMax) {}
        void process() override;
    };

    template<typename TInputPixel, typename TOutputPixel>
    void PoissonNoise<TInputPixel, TOutputPixel>::process()
    {
        // Compute input image maximum
        TInputPixel maxValue{ this->m_inputImage(0, 0) };
        for (int i{}; i < this->m_inputImage.getWidth(); ++i)
            for (int j{}; j < this->m_inputImage.getHeight(); ++j)
                if (maxValue < this->m_inputImage(i, j)) 
                    maxValue = this->m_inputImage(i, j);
        
        // Declare a Poisson distribution
        std::random_device rd;
        std::mt19937 generator(rd());
        using distribution_type = std::poisson_distribution<int>;
        distribution_type distribution(1.0);
        using param_t = distribution_type::param_type;

        // Image traversal to replace pixel values by Poisson distribution samples
        for (int i{}; i < this->m_outputImage.getWidth(); ++i)
            for (int j{}; j < this->m_outputImage.getHeight(); ++j)
            {
                auto poissonMean{ static_cast<int>(this->m_inputImage(i, j) * m_nbPhotonMax / maxValue) };
                if (poissonMean == 0)
                {
                    this->m_outputImage(i, j) = 0;
                }
                else
                {
                    distribution.param(static_cast<param_t>(poissonMean));
                    this->m_outputImage(i, j) = static_cast<TOutputPixel>(distribution(generator));
                }
            }
    }


}

namespace v3_0
{
    using namespace v2_0;
}

#endif
