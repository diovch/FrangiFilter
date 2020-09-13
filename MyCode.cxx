 
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
 
#include "itkRecursiveGaussianImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "eig3.h"

#include <Windows.h>

using namespace std;
#include <math.h>

#ifdef MAX
#undef MAX
#endif

#define MAX(a, b) ((a)>(b)?(a):(b))

#define n 3

static double hypot2(double x, double y) {
	return sqrt(x*x + y * y);
}

// Symmetric Householder reduction to tridiagonal form.
static void tred2(double V[n][n], double d[n], double e[n]);

// Symmetric tridiagonal QL algorithm.
static void tql2(double V[n][n], double d[n], double e[n]);
void SwapVectors(double V[n][n], int left, int right);
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

				// в плоскости или срезе zb
				auto xb_yc_zb = pointer + ((i - 1) + j * width + (k - 1) * width*height);
				auto xc_yb_zb = pointer + (i + (j - 1) * width + (k - 1) * width*height);
				auto xc_yc_zb = pointer + (i + j * width + (k - 1) * width*height);
				auto xc_yf_zb = pointer + (i + (j + 1) * width + (k - 1) * width*height);
				auto xf_yc_zb = pointer + ((i + 1) + j * width + (k - 1) * width*height);

				// в плоскости или срезе z
				auto xb_yb_zc = pointer + ((i - 1) + (j - 1) * width + k * width*height);
				auto xb_yc_zc = pointer + ((i - 1) + j * width + k * width*height);
				auto xb_yf_zc = pointer + ((i - 1) + (j + 1) * width + k * width*height);
				auto xc_yb_zc = pointer + (i + (j - 1) * width + k * width*height);
				auto xc_yc_zc = pointer + (i + j * width + k * width*height);
				auto xc_yf_zc = pointer + (i + (j + 1) * width + k * width*height);
				auto xf_yb_zc = pointer + ((i + 1) + (j - 1) * width + k * width*height);
				auto xf_yc_zc = pointer + ((i + 1) + j * width + k * width*height);
				auto xf_yf_zc = pointer + ((i + 1) + (j + 1) * width + k * width*height);

				// в плоскости или срезе z
				auto xb_yc_zf = pointer + ((i - 1) + j * width + (k + 1) * width*height);
				auto xc_yb_zf = pointer + (i + (j - 1) * width + (k + 1) * width*height);
				auto xc_yc_zf = pointer + (i + j * width + (k + 1) * width*height);
				auto xc_yf_zf = pointer + (i + (j + 1) * width + (k + 1) * width*height);
				auto xf_yc_zf = pointer + ((i + 1) + j * width + (k + 1) * width*height);

				// вторые производные по одной оси
				auto df_xx = (*xf_yc_zc - 2 * (*xc_yc_zc) + *xb_yc_zc) / x_sc / x_sc; 
				auto df_yy = (*xc_yf_zc - 2 * (*xc_yc_zc) + *xc_yb_zc) / y_sc / y_sc;
				auto df_zz = (*xc_yc_zf - 2 * (*xc_yc_zc) + *xc_yc_zb) / z_sc / z_sc;

				// смешанные производные равны
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
				
				/*std::cout << "Eigenvalues: " << d[0] << " " << d[1] << " " << d[2] << std::endl;

				std::cout << "Eigenvector 0: " << V[0][0] << " " << V[1][0] << " " << V[2][0] << std::endl;
				std::cout << "Eigenvector 1: " << V[0][1] << " " << V[1][1] << " " << V[2][1] << std::endl;
				std::cout << "Eigenvector 2: " << V[0][2] << " " << V[1][2] << " " << V[2][2] << std::endl;*/

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
				// заполнение центрального вокселя результирующего файла
				auto centr_res_voxel = result_pointer + (i + j * width + k * width*height);
				*centr_res_voxel = (short)(ves_fun * 1024.0f);
				//*centr_res_voxel = i;
			}
		}
	}

///////////////////////////////////////	запись файла ///////////////////////////////////////
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


/* Eigen decomposition code for symmetric 3x3 matrices, copied from the public
   domain Java Matrix library JAMA. */


static void tql2(double V[n][n], double d[n], double e[n]) {

 //  This is derived from the Algol procedures tql2, by
 //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
 //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
 //  Fortran subroutine in EISPACK.

 for (int i = 1; i < n; i++) {
	 e[i - 1] = e[i];
 }
 e[n - 1] = 0.0;

 double f = 0.0;
 double tst1 = 0.0;
 double eps = pow(2.0, -52.0);
 for (int l = 0; l < n; l++) {

	 // Find small subdiagonal element

	 tst1 = MAX(tst1, fabs(d[l]) + fabs(e[l]));
	 int m = l;
	 while (m < n) {
		 if (fabs(e[m]) <= eps * tst1) {
			 break;
		 }
		 m++;
	 }

	 // If m == l, d[l] is an eigenvalue,
	 // otherwise, iterate.

	 if (m > l) {
		 int iter = 0;
		 do {
			 iter = iter + 1;  // (Could check iteration count here.)

			 // Compute implicit shift

			 double g = d[l];
			 double p = (d[l + 1] - g) / (2.0 * e[l]);
			 double r = hypot2(p, 1.0);
			 if (p < 0) {
				 r = -r;
			 }
			 d[l] = e[l] / (p + r);
			 d[l + 1] = e[l] * (p + r);
			 double dl1 = d[l + 1];
			 double h = g - d[l];
			 for (int i = l + 2; i < n; i++) {
				 d[i] -= h;
			 }
			 f = f + h;

			 // Implicit QL transformation.

			 p = d[m];
			 double c = 1.0;
			 double c2 = c;
			 double c3 = c;
			 double el1 = e[l + 1];
			 double s = 0.0;
			 double s2 = 0.0;
			 for (int i = m - 1; i >= l; i--) {
				 c3 = c2;
				 c2 = c;
				 s2 = s;
				 g = c * e[i];
				 h = c * p;
				 r = hypot2(p, e[i]);
				 e[i + 1] = s * r;
				 s = e[i] / r;
				 c = p / r;
				 p = c * d[i] - s * g;
				 d[i + 1] = h + s * (c * g + s * d[i]);

				 // Accumulate transformation.

				 for (int k = 0; k < n; k++) {
					 h = V[k][i + 1];
					 V[k][i + 1] = s * V[k][i] + c * h;
					 V[k][i] = c * V[k][i] - s * h;
				 }
			 }
			 p = -s * s2 * c3 * el1 * e[l] / dl1;
			 e[l] = s * p;
			 d[l] = c * p;

			 // Check for convergence.

		 } while (fabs(e[l]) > eps*tst1);
	 }
	 d[l] = d[l] + f;
	 e[l] = 0.0;
 }

 // Sort eigenvalues and corresponding vectors.

 for (int i = 0; i < n - 1; i++) {
	 int k = i;
	 double p = d[i];
	 for (int j = i + 1; j < n; j++) {
		 if (d[j] < p) {
			 k = j;
			 p = d[j];
		 }
	 }
	 if (k != i) {
		 d[k] = d[i];
		 d[i] = p;
		 for (int j = 0; j < n; j++) {
			 p = V[j][i];
			 V[j][i] = V[j][k];
			 V[j][k] = p;
		 }
	 }
 }
}


static void tred2(double V[n][n], double d[n], double e[n]) {

	//  This is derived from the Algol procedures tred2 by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for (int j = 0; j < n; j++) {
		d[j] = V[n - 1][j];
	}

	// Householder reduction to tridiagonal form.

	for (int i = n - 1; i > 0; i--) {

		// Scale to avoid under/overflow.

		double scale = 0.0;
		double h = 0.0;
		for (int k = 0; k < i; k++) {
			scale = scale + fabs(d[k]);
		}
		if (scale == 0.0) {
			e[i] = d[i - 1];
			for (int j = 0; j < i; j++) {
				d[j] = V[i - 1][j];
				V[i][j] = 0.0;
				V[j][i] = 0.0;
			}
		}
		else {

			// Generate Householder vector.

			for (int k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			double f = d[i - 1];
			double g = sqrt(h);
			if (f > 0) {
				g = -g;
			}
			e[i] = scale * g;
			h = h - f * g;
			d[i - 1] = f - g;
			for (int j = 0; j < i; j++) {
				e[j] = 0.0;
			}

			// Apply similarity transformation to remaining columns.

			for (int j = 0; j < i; j++) {
				f = d[j];
				V[j][i] = f;
				g = e[j] + V[j][j] * f;
				for (int k = j + 1; k <= i - 1; k++) {
					g += V[k][j] * d[k];
					e[k] += V[k][j] * f;
				}
				e[j] = g;
			}
			f = 0.0;
			for (int j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			double hh = f / (h + h);
			for (int j = 0; j < i; j++) {
				e[j] -= hh * d[j];
			}
			for (int j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (int k = j; k <= i - 1; k++) {
					V[k][j] -= (f * e[k] + g * d[k]);
				}
				d[j] = V[i - 1][j];
				V[i][j] = 0.0;
			}
		}
		d[i] = h;
	}

	// Accumulate transformations.

	for (int i = 0; i < n - 1; i++) {
		V[n - 1][i] = V[i][i];
		V[i][i] = 1.0;
		double h = d[i + 1];
		if (h != 0.0) {
			for (int k = 0; k <= i; k++) {
				d[k] = V[k][i + 1] / h;
			}
			for (int j = 0; j <= i; j++) {
				double g = 0.0;
				for (int k = 0; k <= i; k++) {
					g += V[k][i + 1] * V[k][j];
				}
				for (int k = 0; k <= i; k++) {
					V[k][j] -= g * d[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			V[k][i + 1] = 0.0;
		}
	}
	for (int j = 0; j < n; j++) {
		d[j] = V[n - 1][j];
		V[n - 1][j] = 0.0;
	}
	V[n - 1][n - 1] = 1.0;
	e[0] = 0.0;
}

 void eigen_decomposition(double A[n][n], double V[n][n], double d[n]) {
	 double e[n];
	 for (int i = 0; i < n; i++) {
		 for (int j = 0; j < n; j++) {
			 V[i][j] = A[i][j];
		 }
	 }
	 tred2(V, d, e);
	 tql2(V, d, e);
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