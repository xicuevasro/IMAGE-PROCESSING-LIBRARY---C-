#ifndef __Image_h
#define __Image_h

#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <functional>
#include <random>
#include <algorithm>

const double pi{ std::acos(-1.0) };

namespace v1_0
{
    template<typename T>
    using image_type = std::vector<std::vector<T>>;

    template<typename T>
    struct RGB {
        T R;
        T G;
        T B;
    };

    template<typename T>
    image_type<T> allocate(int width, int height)
    {
        image_type<T> image(width);
        for (int i{}; i < width; ++i)
            image[i].resize(height);
        return image;
    }

    // White image
    template<typename T>
    void white(image_type<T>& image)
    {
        for (std::size_t i{}; i < image.size(); ++i)
            for (std::size_t j{}; j < image[0].size(); ++j)
                image[i][j] = std::numeric_limits<T>::max();
    }

    // Sinusoidal image
    template<typename T>
    void sinus(image_type<T>& image, int n = 1)
    {
        for (std::size_t i{}; i < image.size(); ++i)
        {
            T value{ static_cast<double>(std::sin(2 * pi * i * n / image.size())) };
            for (std::size_t j{}; j < image[0].size(); ++j)
                image[i][j] = value;
        }
    }

    // Checkerboard image
    template<typename T>
    void checkerboard(image_type<T>& image, int nx, int ny)
    {
        auto minOut{ std::numeric_limits<T>::min() }, maxOut{ std::numeric_limits<T>::max() };
        auto sizeX{ image.size() / nx }, sizeY{ image[0].size() / ny };
        for (std::size_t i{}; i < image.size(); ++i)
        {
            auto cellX{ i / sizeX };
            for (std::size_t j{}; j < image[0].size(); ++j)
            {
                auto cellY{ j / sizeY };
                image[i][j] = ((cellX % 2 == 0 && cellY % 2 == 0)
                            || (cellX % 2 != 0 && cellY % 2 != 0))
                            ? maxOut : minOut;
            }
        }
    }

    // Conversion between image types T and U
    template<typename T, typename U>
    image_type<U> convert(const image_type<T>& imageIn, bool rescale = false)
    {
        auto imageOut{ allocate<U>(imageIn.size(), imageIn[0].size()) };

        // Adjust output range to input range
        if (rescale)
        {
            // Compute minimum and maximum input gray levels
            auto inMin{ imageIn[0][0] }, inMax{ inMin };
            for (std::size_t i{}; i < imageIn.size(); ++i)
                for (std::size_t j{}; j < imageIn[0].size(); ++j)
                {
                    if (imageIn[i][j] > inMax)
                        inMax = imageIn[i][j];
                    else if (imageIn[i][j] < inMin)
                        inMin = imageIn[i][j];
                }
            auto outMin{ std::numeric_limits<U>::min() };
            auto outMax{ std::numeric_limits<U>::max() };
            for (std::size_t i{}; i < imageIn.size(); ++i)
                for (std::size_t j{}; j < imageIn[0].size(); ++j)
                    imageOut[i][j] = static_cast<U>(outMin
                        + (outMax - outMin) * (imageIn[i][j] - inMin) / static_cast<double>(inMax - inMin));
        }
        else
        {
            for (std::size_t i{}; i < imageIn.size(); ++i)
                for (std::size_t j{}; j < imageIn[0].size(); ++j)
                    imageOut[i][j] = static_cast<U>(imageIn[i][j]);
        }

        return imageOut;
    }

    // Convert a gray level image of type T to a RGB one
    template<typename T>
    image_type<RGB<T>> convert2RGB(const image_type<T>& image, const std::string& lutName, bool binary = true)
    {
        const auto numGrayLevel{ static_cast<std::size_t>(std::pow(256, sizeof(T))) };
        std::vector<RGB<T>> lut(numGrayLevel);

        if (binary)
        {
            std::ifstream file(lutName, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Can't open binary LUT file " << lutName << '\n';
                return allocate<RGB<T>>(0, 0);
            }

            file.seekg(32, std::ios::cur); // 32 bytes offset from current position
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].R), sizeof(T));
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].G), sizeof(T));
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].B), sizeof(T));
            file.close();
        }
		else
        {
            std::ifstream file(lutName);
            if (!file.is_open())
            {
                std::cerr << "Can't open binary LUT file " << lutName << '\n';
                return allocate<RGB<T>>(0,0);
            }
            std::string line;
            std::getline(file, line);
            for (std::size_t i{}; i < numGrayLevel; ++i)
            {
                std::getline(file, line);
                std::istringstream iss(line);
                std::string item;
                std::getline(iss, item, '\t');
                std::getline(iss, item, '\t');
                lut[i].R = static_cast<T>(std::stoi(item));
                std::getline(iss, item, '\t');
                lut[i].G = static_cast<T>(std::stoi(item));
                std::getline(iss, item, '\t');
                lut[i].B = static_cast<T>(std::stoi(item));
            }
            file.close();
        }

        auto imageRGB{ allocate<RGB<T>>(image.size(), image[0].size()) };
        for (std::size_t i{}; i < image.size(); ++i)
            for (std::size_t j{}; j < image[0].size(); ++j)
            {
                const auto grayLevel{ static_cast<std::size_t>(image[i][j]) };
                imageRGB[i][j] = lut[grayLevel];
            }

        return imageRGB;
    }

    template<typename T>
    void swap(T* p)
    {
        auto q{ reinterpret_cast<char*>(p) };
        std::reverse(q, q + sizeof(T));
    }

    // Read raw image from file
    template<typename T>
    image_type<T> read(const std::string& fileName, int width, int height, bool littleEndian = true)
    {
        std::ifstream file(fileName, std::ios::binary);

        // Error message if cannot open file
        if (!file.is_open())
        {
            std::cerr << "Can't open file " << fileName << '\n';
            return allocate<T>(0, 0);
        }

        auto image{ allocate<T>(width, height) };

        for (int j{}; j < height; ++j)
            for (int i{}; i < width; ++i)
            {
                T value{};
                file.read(reinterpret_cast<char*>(&value), sizeof(T));
                if (littleEndian) swap(&value);
                image[i][j] = value;
            }

        file.close();

        return image;
    }

    // Write raw image to file
    template<typename T>
    void write(image_type<T>& image, const std::string& fileName)
    {
        std::ofstream file(fileName, std::ios_base::binary);
        for (std::size_t j{}; j < image[0].size(); ++j)
            for (std::size_t i{}; i < image.size(); ++i)
                file.write(reinterpret_cast<char*>(&image[i][j]), sizeof(T));
        file.close();
    }

}

namespace v1_1
{
    const double pi{ std::acos(-1.0) };

    template <typename T>
    class Image
    {
        using image_type = std::vector< std::vector< T > >;

    protected:
        image_type m_pixels;
        std::size_t m_width, m_height;

    public:
        Image(std::size_t width = 0, std::size_t height = 0);

        T& operator()(int i, int j) { return m_pixels[i][j]; }
        const T& operator()(int i, int j) const { return m_pixels[i][j]; }

        int getWidth() { return m_width; }
        int getWidth() const { return m_width; }
        int getHeight() { return m_height; }
        int getHeight() const { return m_height; }

        void white();
        void sinus(int n = 1);
        void checkerboard(int nx, int ny);
        void fill(const std::function<double(int,int)>& f);
        void fill(const std::vector<T>& v);
		void randomEllipses(const int n);

        Image<T> swap();

        template<typename U>
        Image<U> convert(bool rescale = false);

        bool read(const std::string& fileName, bool littleEndian = true);
        void write(const std::string& fileName);
    };

    template<typename T>
    Image<T>::Image(std::size_t width, std::size_t height)
        : m_width(width), m_height(height)
    {
        m_pixels = image_type(width);
        for (std::size_t i{ 0 }; i < width; ++i)
            m_pixels[i] = std::vector<T>(height);
    }

    template<typename T>
    void Image<T>::white()
    {
        for (std::size_t i{ 0 }; i < m_width; ++i)
            for (std::size_t j{ 0 }; j < m_height; ++j)
                m_pixels[i][j] = std::numeric_limits<T>::max();
    }

    template<typename T>
    void Image<T>::sinus(int n)
    {
        for (std::size_t i{ 0 }; i < m_width; ++i)
        {
            T value{ static_cast<T>(std::sin(2 * pi * n * i / (m_width - 1))) };
            for (std::size_t j{ 0 }; j < m_height; ++j)
                m_pixels[i][j] = value;
        }
    }

    template<typename T>
    void Image<T>::checkerboard(int nx, int ny)
    {
        auto minOut{ std::numeric_limits<T>::min() }, maxOut{ std::numeric_limits<T>::max() };
        auto sizeX{ m_width / nx }, sizeY{ m_height / ny };
        for (std::size_t i{}; i < m_width; ++i)
        {
            auto cellX{ i / sizeX };
            for (std::size_t j{}; j < m_height; ++j)
            {
                auto cellY{ j / sizeY };
                m_pixels[i][j] = ((cellX % 2 == 0 && cellY % 2 == 0)
                            || (cellX % 2 != 0 && cellY % 2 != 0))
                            ? maxOut : minOut;
            }
        }
    }

    template<typename T>
    void Image<T>::fill(const std::function<double(int,int)>& f)
    {
        const auto nx{ m_width / 2 }, ny{ m_height / 2 };
        for (std::size_t i{}; i < m_width; ++i)
            for (std::size_t j{}; j < m_height; ++j)
                m_pixels[i][j] = static_cast<T>(f(i - nx, j - ny));
    }

    template<typename T>
    void Image<T>::fill(const std::vector<T>& v)
    {
        int k{};
        for (std::size_t j{}; j < m_height; ++j)
            for (std::size_t i{}; i < m_width; ++i)
                m_pixels[i][j] = v[k++];
    }

	template<typename T>
    void Image<T>::randomEllipses(const int n)
    {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distribx(0, m_width), distriby(0, m_height);

        for (int k{ 0 }; k < n; ++k)
        {
            auto rx{ m_width / 8 + distribx(gen) / 8 };
            auto ry{ m_height / 8 + distriby(gen) / 8 };
            auto x0{ distribx(gen) };
            auto y0{ distriby(gen) };
            for (std::size_t i{ 0 }; i < m_width; ++i)
                for (std::size_t j{ 0 }; j < m_height; ++j)
                {
                    auto dx{ static_cast<double>(i) - x0 }, dy{ static_cast<double>(j) - y0 };
                    if ( dx * dx / rx / rx + dy * dy / ry / ry < 1.0 )
                        this->m_pixels[i][j] =  static_cast<T>(k + 1);
                }
        }
    }

    // Swap axes
    template<typename T>
    Image<T> Image<T>::swap()
    {
        Image<T> imageOut(m_height, m_width);
        for (std::size_t i{}; i < m_width; ++i)
                for (std::size_t j{}; j < m_height; ++j)
                    imageOut(j, i) = m_pixels[i][j];
        return imageOut;
    }

    // T type de codage de l'image d'entrée, U type de codage de l'image de sortie
    // Ajustement de dynamique possible si rescale = true
    template<typename T>
    template<typename U>
    Image<U> Image<T>::convert(bool rescale)
    {
        Image<U> imageOut(m_width, m_height);

        // Adjust output range to input range
        if (rescale)
        {
            // Compute minimum and maximum input gray levels
            auto inMin{ m_pixels[0][0] }, inMax{ inMin };
            for (std::size_t i{}; i < m_width; ++i)
                for (std::size_t j{}; j < m_height; ++j)
                {
                    if (m_pixels[i][j] > inMax)
                        inMax = m_pixels[i][j];
                    else if (m_pixels[i][j] < inMin)
                        inMin = m_pixels[i][j];
                }

            auto outMin{ std::numeric_limits<U>::min() };
            auto outMax{ std::numeric_limits<U>::max() };
            for (std::size_t i{}; i < m_width; ++i)
                for (std::size_t j{}; j < m_height; ++j)
                    imageOut(i, j) = static_cast<U>(outMin
                        + (outMax - outMin) * (m_pixels[i][j] - inMin) / static_cast<double>(inMax - inMin));
        }
        else
        {
            for (std::size_t i{}; i < m_width; ++i)
                for (std::size_t j{}; j < m_height; ++j)
                    imageOut(i, j) = static_cast<U>(m_pixels[i][j]);
        }

        return imageOut;
    }

    // Swap byte order
    template <typename T>
    void endswap(T *objp)
    {
        unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
        std::reverse(memp, memp + sizeof(T));
    }

    // T type de codage du fichier
    template<typename T>
    bool Image<T>::read(const std::string& fileName, bool littleEndian)
    {
        std::ifstream fIn(fileName, std::ios::in | std::ios::binary);
        if (!fIn.is_open())
        {
            std::cerr << "Can't open " << fileName << '\n';
            return false;
        }

        for (std::size_t j{}; j < m_height; ++j)
            for (std::size_t i{}; i < m_width; ++i)
            {
                T value{};
                fIn.read( reinterpret_cast<char*>(&value), sizeof(T) );
                if (littleEndian) endswap(&value);
                m_pixels[i][j] = value;
            }

        fIn.close();

        return true;
    }

    template<typename T>
    void Image<T>::write(const std::string& fileName)
    {
        std::ofstream fOut(fileName, std::ios::out | std::ios::binary);
        if (!fOut.is_open())
        {
            std::cerr << "Can't open " << fileName << '\n';
            return;
        }

        for (std::size_t j{}; j < m_height; ++j)
            for (std::size_t i{}; i < m_width; ++i)
                fOut.write( reinterpret_cast<char*>(&m_pixels[i][j]), sizeof(T));

        fOut.close();
    }

    template <typename T>
    class RGB
    {
    public:
        T R, G, B;
        T& operator[](int i) {
            if (i == 0) return R;
            else if (i == 1) return G;
            else return B;
        }
        const T& operator[](int i) const {
            if (i == 0) return R;
            else if (i == 1) return G;
            else return B;
        }
        const int size() { return 3; }
    };

	template <typename T>
    class ImageRGB : public Image<RGB<T>>
    {
    public:
        ImageRGB(int width = 0, int height = 0) : Image<RGB<T>>(width, height) {}
        ImageRGB(const Image<T>& image, const std::string& lutName, bool binary = true);
    };

    template<typename T>
    ImageRGB<T>::ImageRGB(const Image<T>& image, const std::string& lutName, bool binary)
        : Image<RGB<T>>(image.getWidth(), image.getHeight())
    {
        const auto numGrayLevel{ static_cast<std::size_t>(std::pow(256, sizeof(T))) };
        std::vector<RGB<T>> lut(numGrayLevel);

        if (binary)
        {
            std::ifstream file(lutName, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Can't open binary LUT file " << lutName << '\n';
                return;
            }

            file.seekg(32, std::ios::cur); // 32 bytes offset from current position
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].R), sizeof(T));
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].G), sizeof(T));
            for (std::size_t i{}; i < numGrayLevel; ++i)
                file.read(reinterpret_cast<char*>(&lut[i].B), sizeof(T));
            file.close();
        }
		else
        {
            std::ifstream file(lutName);
            if (!file.is_open())
            {
                std::cerr << "Can't open binary LUT file " << lutName << '\n';
                return;
            }
            std::string line;
            std::getline(file, line);
            for (std::size_t i{}; i < numGrayLevel; ++i)
            {
                std::getline(file, line);
                std::istringstream iss(line);
                std::string item;
                std::getline(iss, item, '\t');
                std::getline(iss, item, '\t');
                lut[i].R = static_cast<T>(std::stoi(item));
                std::getline(iss, item, '\t');
                lut[i].G = static_cast<T>(std::stoi(item));
                std::getline(iss, item, '\t');
                lut[i].B = static_cast<T>(std::stoi(item));
            }
            file.close();
        }

        for (std::size_t i{ 0 }; i < this->m_width; ++i)
            for (std::size_t j{ 0 }; j < this->m_height; ++j)
            {
                const auto index{ static_cast<std::size_t>(image(i,j)) };
                this->m_pixels[i][j] = lut[index];
            }
    }
}

namespace v2_0
{
    using namespace v1_1;
}

#endif
