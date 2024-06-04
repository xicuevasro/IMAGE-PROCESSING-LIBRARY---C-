#include <iostream>

#include "include\Image.h"
#include "include\processing.h"
#include "include\reconstruction.h"

#pragma warning( disable : 4100 4101 4127 4181 4211 4244 4273 4324 4503 4512 4522 4700 4714 4717 4800 5054)
#include "Eigen/Dense"

int main()
{
    using input_pixel_type = unsigned char;
    using pixel_type = double;
    using output_pixel_type = unsigned char;

    constexpr int width{ 256 }, height{ 256 };

    /**************************
     ***** TECSANLIB V1_0 *****
     **************************/

    // Create a new image
    //auto img{ v1_0::allocate<pixel_type>(width, height) };
    //v1_0::white(img);
    //v1_0::sinus(img, 8);
    //v1_0::checkerboard(img, 8, 8);

    // Read input image from raw file
    //auto imgIn{ v1_0::read<input_pixel_type>("./images/TEMP_16_bits_64x64-coeur.raw", width, height) };

    // Convert to processing type
    //auto img{ v1_0::convert<input_pixel_type, pixel_type>(imgIn) };

    // Processing...

    // Convert image to output type
    //auto imgOut{ v1_0::convert<pixel_type, output_pixel_type>(img, true) };

    // Write image to output raw file
    //v1_0::write(imgOut, "output.raw");

    // Convert output image to RGB image using binary LUT
    //auto imgRGB{ v1_0::convert2RGB(imgOut, "./LUT/NucMed_Image_LUTs/Fire-1.lut") };
    //auto imgRGB{ v1_0::convert2RGB(imgOut, "./LUT/001-fire.lut", false) };
    //v1_0::write(imgRGB, "imageRGB.raw");

    /**************************
     ***** TECSANLIB V1_1 *****
     **************************/
/*
    // Create a new image
    v1_1::Image<input_pixel_type> imgIn(width, height);

    // Read the image content from raw file
    if (!(imgIn.read("./images/TDM_8_bits_512x512_thorax.raw")))
        return -1;

    // Convert to processing type
    auto img{ imgIn.convert<pixel_type>() };

    // Convert to output file type
    auto imgOut{ img.convert<output_pixel_type>() };

    // Write to raw file
    imgOut.write("output.raw");

    // Convert to RGB image
    v1_1::ImageRGB<output_pixel_type> imgRGB(imgOut, "./LUT/NucMed_Image_LUTs/Fire-1.lut");
    imgRGB.write("imageRGB.raw");
*/

    /**************************
     ***** TECSANLIB V2_0 *****
     **************************/

    // Create two new images
    //v2_0::Image<input_pixel_type> imgIn1(width, height);
    //if (!(imgIn1.read("./images/TDM_16_bits_512x512_crane.raw")))
    //    return -1;

    //v2_0::Image<input_pixel_type> imgIn2(width, height);
    //imgIn2.checkerboard(8, 8);

    // Convert to processing type
    //auto img1{ imgIn1.convert<pixel_type>() };
    //auto img2{ imgIn2.convert<pixel_type>() };

    // Addition processing (class or function version)
    //v2_0::Addition<pixel_type> addition(img1, img2);
    //addition.update();
    //auto img{ addition.getOutput() };
    //auto img{ v2_0::addition<pixel_type>(img1, img2) };

    // Scalar addition processing (class or function version)
    //v2_0::AdditionScalar<pixel_type> additionScalar(img1, 12, true);
    //additionScalar.update();
    //auto img{ additionScalar.getOutput() };
    //auto img{ v2_0::additionScalar<pixel_type>(img1, 12) };

    // Equalization (class version)
    //auto minOut{ std::numeric_limits<pixel_type>::min() };
    //auto maxOut{ std::numeric_limits<pixel_type>::max() };
    //v2_0::Equalization<pixel_type> equalization(img1, minOut, maxOut);
    //equalization.update();
    //auto img{ equalization.getOutput() };

    // Convolution (spatial filtering)
    //const int kernelSize{ 9 };
    //v2_0::Image<pixel_type> kernel(kernelSize, kernelSize);
    // Mean filter
    //kernel.fill([kernelSize](int i, int j) { return 1. / (kernelSize * kernelSize); });
    // Gaussian filter
    //constexpr auto sigma { 2.0 };
    //const auto sigma2{ sigma * sigma };
    //auto gauss{
    //    [sigma2](int i, int j)
    //    {
    //        return 0.5 * std::exp(-0.5 * (i * i + j * j) / sigma2) / pi / sigma2;
    //    }
    //};
    //kernel.fill(gauss);
    // Exponential filter
    //constexpr auto beta{ 0.1 };
    //auto exponential{
    //    [beta](int i, int j)
    //    {
    //        return 0.25 * beta * beta * std::exp(-beta * (std::abs(i) + std::abs(j)));
    //    }
    //};
    //kernel.fill(exponential);
    //kernel.convert<output_pixel_type>(true).write("kernel.raw");
    //auto img{ v2_0::convolution(img1, kernel) };

    // Frequential filtering
    //v2_0::Image<pixel_type> mtf(img1.getWidth(), img1.getHeight());
    //constexpr auto cuttingFrequency{ 0.25 };
    // Ideal filter
    //auto ideal{
    //    [cuttingFrequency, width, height](int i, int j)
    //    {
    //        auto u{ i / cuttingFrequency / width }, v{ j / cuttingFrequency / height };
    //        return (u * u + v * v <= 1.0);
    //    }
    //};
    //mtf.fill(ideal);

    // Butterworth filter
    // constexpr auto n{ 3 };
    // auto butterworth{
    //     [cuttingFrequency, n, width, height](int i, int j)
    //     {
    //         auto u{ i / cuttingFrequency / width }, v{ j / cuttingFrequency / height };
    //         auto f_f02{ u * u + v * v };
    //         double tmp{ f_f02 };
    //         for (int k{ 1 }; k < n; ++k) tmp *= f_f02;
    //         return 1. / ( 1. + (std::sqrt(2) - 1) * tmp);
    //     }
    // };
    // mtf.fill(butterworth);

    // mtf.convert<output_pixel_type>(true).write("mtf.raw");
    // auto img{ v2_0::frequentialFiltering(img1, mtf) };

    // Contour detection using Sobel kernels
    // // X derivative
    // v2_0::Image<pixel_type> sobelX(3, 3);
    // sobelX.fill({-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0});
    // auto dX{ v2_0::convolution(img1, sobelX) };
    // // Y derivative
    // v2_0::Image<pixel_type> sobelY(3, 3);
    // sobelY.fill({-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0});
    // auto dY{ v2_0::convolution(img1, sobelY) };
    // // Gradient norm
    // auto norm{ v2_0::norm(dX, dY) };

    // pixel_type min{img1(0, 0)};
    // for (int i{}; i < img1.getWidth(); ++i)
    //     for (int j{}; j < img1.getHeight(); ++j)
    //         if (img1(i, j) < min)
    //             min = img1(i, j);

    // Multi thresholding
    // const std::vector<pixel_type> thresholds{ -1100, -100, 0, 100 };
    // v2_0::MultiThresholding<pixel_type> multiThresholding(img1, thresholds);
    // multiThresholding.update();
    // auto img{ multiThresholding.getOutput() };

    // Classification using k-means algorithm
    //constexpr int nClasses{ 6 }, nDim{ 4 };
    //v2_0::Image<pixel_type> groundTruth(width, height);
    //if (!groundTruth.read("./Images/Ellipse_5_labels_8_bits_256x256.raw"))
    //if (!(groundTruth.read("./Images/IRM_5_labels_8_bits_256x256_crane.raw", false)))
    //    return -1;
    //groundTruth.randomEllipses(nClasses - 1);
    //auto imgIn1{ v2_0::gaussianDistributions<pixel_type, pixel_type>(groundTruth, nDim, 2.0) };

    // k-means classification
    //v2_0::KMeans<input_pixel_type, pixel_type> kMeans(imgIn1, nClasses);
    //kMeans.update();
    //auto img{ kMeans.getOutput() };

    // Display agreement matrix
    //kMeans.concordanceMatrix(std::cout, groundTruth);

    // Convert image to output type
    //auto imgOutGroundTruth{ groundTruth.convert<output_pixel_type>(true) };
    //auto imgOut{ img.convert<output_pixel_type>(true) };

    // Write image to raw file
    //imgOutGroundTruth.write("groundTruth.raw");
    //imgOut.write("kmeans.raw");


    /**************************
     ***** TECSANLIB V3_0 *****
     **************************/
/*
    // Eigen matrix
    Eigen::MatrixXd m1(2, 2);
    //Eigen::Matrix2d m;
    m1(0, 0) = 1.0;
    m1(0, 1) = 2.0;
    m1(1, 0) = 3.0;
    m1(1, 1) = 4.0;
    std::cout << "m1=\n" << m1 << '\n';
    auto m2{ Eigen::MatrixXd::Constant(2, 2, 1.0) };
    std::cout << "m2=\n" << m2 << '\n';
    auto m3{ Eigen::MatrixXd::Random(3, 3) };
    std::cout << "m3=\n" << m3 << '\n';
    Eigen::Matrix2d m4{
        { 1.0, 2.0 },
        { 3.0, 4.0 }
    };
    std::cout << "m4=\n" << m4 << '\n';

    // Eigen vector
    Eigen::VectorXd v1(3);
    v1(0) = 1.0;
    v1(1) = 2.0;
    v1(2) = 3.0;
    std::cout << "v1=\n" << v1 << '\n';
    Eigen::Vector3d v2;
    v2 << 1.0, 2.0, 3.0;
    std::cout << "v2=\n" << v2 << '\n';

    // Matrix / vector operations
    std::cout << "m3 * v1 =\n" << m3 * v1 << '\n';

    // Matrix block
    constexpr int n_rows{ 6 }, n_cols{ 6 };
    auto m5{ Eigen::MatrixXd(n_rows, n_cols) };
    for (int i{}, k{}; i < n_rows; ++i)
        for (int j{}; j < n_cols; ++j)
            m5(i, j) = ++k;
    std::cout << "m5=\n" << m5 << '\n';
    // Extract a 3x3 block at location (2,2)
    auto m6{ m5.block<3, 3>(2, 2) };    
    std::cout << "m6=\n" << m6 << '\n';

    // System solving
    Eigen::Matrix3d m7;
    m7 << 1,2,3, 4,5,6, 7,8,10;
    std::cout << "m7=\n" << m7 << '\n';
    Eigen::Vector3d v3;
    v3 << 3,3,4;
    std::cout << "b=\n" << v3 << '\n';
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> solver(m7);
    auto x1{ solver.solve(v3) };
    std::cout << "x1=\n" << x1 << '\n';

    // System solving for symmetric matrices
    Eigen::Matrix2f m8, v4;
    m8 << 2, -1, -1, 3;
    v4 << 1, 2, 3, 1;
    std::cout << "m8=\n" << m8 << std::endl;
    std::cout << "v4=\n" << v4 << std::endl;
    Eigen::Matrix2f x2 { m8.ldlt().solve(v4) };
    std::cout << "x2=\n" << x2 << std::endl;
*/

    constexpr int nProj{ 128 }, nRays{ 256 };
    const int imageSize{ width };
    const double angleRange{ pi };

    //v3_0::ParallelProjection parallelProjection(nProj, nRays, angleRange, imageSize);
    //parallelProjection.computeRays();

    // Read input image file
    v3_0::Image<input_pixel_type> imgIn(width, height);
    imgIn.read("./images/Shepp_Logan_phantom_8_bits_256x256.raw", false);
    //imgIn.read("./images/Shepp_Logan_sinogram_16_bits_256x256.raw", false);

    // Convert to processing type
    auto img1{ imgIn.convert<pixel_type>() };

    // Sinogram simulation
    std::cout << "Computing sinogram\n";
    v3_0::Sinogram<pixel_type> sinogram(img1, nProj, nRays, angleRange, imageSize);
    sinogram.setIntersectionType(v3_0::ParallelProjection::DIRAC);
    sinogram.update();
    auto img2{ sinogram.getOutput() };

    img2.convert<output_pixel_type>(true).write("sinogram_without_noise.raw");

    // Add Poisson noise to sinogram
    std::cout << "Adding Poisson noise\n";
    constexpr pixel_type nbPhotonMax{ 1000 };
    v3_0::PoissonNoise<pixel_type> poissonNoise(img2, nbPhotonMax, true);
    poissonNoise.update();

    //img2.convert<output_pixel_type>(true).write("sinogram_with_noise.raw");

    // Reconstruction by filtered backprojection
    v3_0::Reconstruction<pixel_type> reconstruction(img2, angleRange, imageSize);
    reconstruction.setReconstructionType(v3_0::Reconstruction<pixel_type>::ART);
    //reconstruction.setFilterType(v3_0::Reconstruction<pixel_type>::BUTTERWORTH);
    //reconstruction.setCutoffFrequency(0.25);
    reconstruction.update();

    // Write ouput image to raw file
    auto out{ reconstruction.getOutput().convert<output_pixel_type>(true) };
    out.write("reconstruction.raw");

    return 0;
}
