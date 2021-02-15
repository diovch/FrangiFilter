 
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
 
#include "itkRecursiveGaussianImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "eig3.h"

#include <Windows.h>

using namespace std;
#include <math.h>

void SwapVectors(double V[n][n], int left, int right);
void SwapValues(double d[n], int left, int right);
void AbsoluteSort(double d[n], double V[n][n]);


int main(int argc, char * argv[])
{
	
	if (argc != 4)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << " <InputImageFile> <OutputImageFile> <sigma>" << std::endl;
		return EXIT_FAILURE;
	}

	constexpr unsigned int Dimension = 3;

	const char * inputFileName = argv[1];
	const char * outputFileName = argv[2];
	const float  sigmaValue = std::stod(argv[3]);

	using PixelType = short;
	using ImageType = itk::Image<PixelType, Dimension>;

	using ReaderType = itk::ImageFileReader<ImageType>;
	ReaderType::Pointer reader = ReaderType::New();
	char curDir[256];
	GetCurrentDirectory(256, curDir);

	reader->SetFileName(inputFileName);
	reader->Update();

	using FilterType = itk::RecursiveGaussianImageFilter<ImageType, ImageType>;
	FilterType::Pointer smoothFilterX = FilterType::New();
	FilterType::Pointer smoothFilterY = FilterType::New();
	FilterType::Pointer smoothFilterZ = FilterType::New();
	smoothFilterX->SetDirection(0);
	smoothFilterY->SetDirection(1);
	smoothFilterZ->SetDirection(2);
	smoothFilterX->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
	smoothFilterY->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
	smoothFilterZ->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
	smoothFilterX->SetNormalizeAcrossScale(true);
	smoothFilterY->SetNormalizeAcrossScale(true);
	smoothFilterZ->SetNormalizeAcrossScale(true);
	smoothFilterX->SetInput(reader->GetOutput());
	smoothFilterY->SetInput(smoothFilterX->GetOutput());
	smoothFilterZ->SetInput(smoothFilterY->GetOutput());
	const double sigma = sigmaValue;
	smoothFilterX->SetSigma(sigma); smoothFilterY->SetSigma(sigma); smoothFilterZ->SetSigma(sigma);
	smoothFilterZ->Update();

	auto image = smoothFilterZ->GetOutput();
	auto pointer = image->GetBufferPointer();
	
	auto region = image->GetBufferedRegion();
	auto size = region.GetSize();
	int width = size[0], height = size[1], depth = size[2];
	
	auto scale = image->GetSpacing();
	auto x_sc = scale[0], y_sc = scale[1], z_sc = scale[2];

	ImageType::Pointer result_image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0; // first index on X
	start[1] = 0; // first index on Y
	start[2] = 0; // first index on Z
	ImageType::SizeType res_size;
	res_size[0] = width; // size along X
	res_size[1] = height; // size along Y
	res_size[2] = depth; // size along Z
	ImageType::RegionType res_region;
	res_region.SetSize(size);
	res_region.SetIndex(start);
	result_image->SetRegions(res_region);
	result_image->SetSpacing(image->GetSpacing());
	result_image->Allocate();
	auto result_pointer = result_image->GetBufferPointer();
	//cout << result_image << endl;
#pragma omp parallel for
	for (int k = 1; k < depth - 1; ++k) {
		for (size_t i = 1; i < size_t(width - 1); ++i) {
			for (size_t j = 1; j < size_t(height - 1); ++j) {
			
				// backward, forward, central

				// voxels in zb slice 
				auto xb_yc_zb = pointer + ((i - 1) + j * width + (k - 1) * width*height);
				auto xc_yb_zb = pointer + (i + (j - 1) * width + (k - 1) * width*height);
				auto xc_yc_zb = pointer + (i + j * width + (k - 1) * width*height);
				auto xc_yf_zb = pointer + (i + (j + 1) * width + (k - 1) * width*height);
				auto xf_yc_zb = pointer + ((i + 1) + j * width + (k - 1) * width*height);

				// voxels in zc slice 
				auto xb_yb_zc = pointer + ((i - 1) + (j - 1) * width + k * width*height);
				auto xb_yc_zc = pointer + ((i - 1) + j * width + k * width*height);
				auto xb_yf_zc = pointer + ((i - 1) + (j + 1) * width + k * width*height);
				auto xc_yb_zc = pointer + (i + (j - 1) * width + k * width*height);
				auto xc_yc_zc = pointer + (i + j * width + k * width*height);
				auto xc_yf_zc = pointer + (i + (j + 1) * width + k * width*height);
				auto xf_yb_zc = pointer + ((i + 1) + (j - 1) * width + k * width*height);
				auto xf_yc_zc = pointer + ((i + 1) + j * width + k * width*height);
				auto xf_yf_zc = pointer + ((i + 1) + (j + 1) * width + k * width*height);

				// voxels in zf slice 
				auto xb_yc_zf = pointer + ((i - 1) + j * width + (k + 1) * width*height);
				auto xc_yb_zf = pointer + (i + (j - 1) * width + (k + 1) * width*height);
				auto xc_yc_zf = pointer + (i + j * width + (k + 1) * width*height);
				auto xc_yf_zf = pointer + (i + (j + 1) * width + (k + 1) * width*height);
				auto xf_yc_zf = pointer + ((i + 1) + j * width + (k + 1) * width*height);

				// second order derivatives along each directions
				auto df_xx = (*xf_yc_zc - 2 * (*xc_yc_zc) + *xb_yc_zc) / x_sc / x_sc; 
				auto df_yy = (*xc_yf_zc - 2 * (*xc_yc_zc) + *xc_yb_zc) / y_sc / y_sc;
				auto df_zz = (*xc_yc_zf - 2 * (*xc_yc_zc) + *xc_yc_zb) / z_sc / z_sc;

				// second order mixed derivatives 
				auto df_yx = (*xf_yf_zc - *xb_yf_zc - *xf_yb_zc + *xb_yb_zc) / 4 / x_sc / y_sc; 
				auto df_yz = (*xc_yf_zf - *xc_yb_zf - *xc_yf_zb + *xc_yb_zb) / 4 / z_sc / y_sc;
				auto df_xz = (*xf_yc_zf - *xf_yc_zb - *xb_yc_zf + *xb_yc_zb) / 4 / z_sc / x_sc;

				double H[3][3]; // Hessian
				H[0][0] = df_xx; H[1][1] = df_yy; H[2][2] = df_zz;
				H[0][1] = H[1][0] = df_yx;
				H[0][2] = H[2][0] = df_xz;
				H[1][2] = H[2][1] = df_yz;

				double V[3][3]; // matrix of eigenvectors
				double d[3];	// vector of eigenvalues
				eigen_decomposition(H, V, d); // solving eigenproblem
				AbsoluteSort(d, V); // sorting in order of incresing of abs of eigenvalues
				
				double lambda1 = d[0], lambda2 = d[1], lambda3 = d[2];
				//The first ratio accounts for the deviation from a blob-like structure

				double Rb = lambda1 / sqrt(lambda2*lambda3); 
				// This ratio is essential for distinguishing between plate-like and line-like structures

				double Ra = fabs(lambda2 / lambda3);
				// second order structureness - Frobenious norm of the Hessian

				double S = sqrt(lambda1*lambda1 + lambda2*lambda2 + lambda3*lambda3);
				// vesselness function

				double const_c = 200; // constant value should be a half of maximum of Hessian norm 

				//vesselness
				double ves_fun = (lambda2 > 0 || lambda3 > 0) ?
					0 : (1 - exp(-Ra * Ra / 0.5))*exp(-Rb * Rb / 0.5)*(1 - exp(-S * S / const_c / const_c));
				

				auto centr_res_voxel = result_pointer + (i + j * width + k * width*height);
				*centr_res_voxel = (short)(ves_fun * 1024.0f);
			}
		}
	}

///////////////////////////////////////	file writing ///////////////////////////////////////
	using WriterType = itk::ImageFileWriter<ImageType>;
	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(result_image);
	writer->SetFileName(outputFileName);

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & error)
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

void SwapValues(double d[n], int left, int right) {
	 double temp = d[left];
	 d[left] = d[right];
	 d[right] = temp;
 }

void SwapVectors(double V[n][n], int left, int right) {
	 for (int i = 0; i < n; ++i) {
		 double temp = V[i][left];
		 V[i][left] = V[i][right];
		 V[i][right] = temp;
	 }
 }

void AbsoluteSort(double d[n], double V[n][n]) {

	 bool swap;
	 do {
		 swap = false;
		 for (int i = 1; i < n; ++i) {
			 if (fabs(d[i]) < fabs(d[i - 1])) {
				 SwapValues(d, i, i - 1);
				 SwapVectors(V, i, i - 1);
				 swap = true;
			 }
		 }
	 } while (swap != false);
}