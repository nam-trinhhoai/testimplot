#include "KohonenOnMultispectralImages.h"
#include "KohonenEngine.h"
#include "cudaimagepaletteholder.h"
#include "imageformats.h"
#include "sampletypebinder.h"
#include "rgblayerfromdataset.h"
#include "fixedlayerfromdataset.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#include <QDebug>

KohonenImage2D::KohonenImage2D() {}
KohonenImage2D::~KohonenImage2D() {}

KohonenCudaPlanarImage2D::KohonenCudaPlanarImage2D(const std::vector<const float*>& stack, std::size_t dimI, std::size_t dimJ, short nonValue) {
	m_stack = stack;
	m_dimI = dimI;
	m_dimJ = dimJ;
	m_nonValue = nonValue;
}

KohonenCudaPlanarImage2D::~KohonenCudaPlanarImage2D() {

}

bool KohonenCudaPlanarImage2D::isExampleValid(long i, long j) const {
	bool ok = i>=0 && i<m_dimI && j>=0 && j<m_dimJ;
	std::size_t index = 0;

	while (ok && index<m_stack.size()) {
		short val = m_stack[index][i+j*m_dimI];
		ok = val!=m_nonValue;
		index++;
	}

	return ok;
}

bool KohonenCudaPlanarImage2D::getExample(float* tab, long i, long j) const {
	bool ok = i>=0 && i<m_dimI && j>=0 && j<m_dimJ;
	std::size_t index = 0;

	while (ok && index<m_stack.size()) {
		short val = m_stack[index][i+j*m_dimI];
		ok = val!=m_nonValue;
		tab[index] = val;
		index++;
	}

	return ok;
}

std::size_t KohonenCudaPlanarImage2D::dimI() const {
	return m_dimI;
}

std::size_t KohonenCudaPlanarImage2D::dimJ() const {
	return m_dimJ;
}

std::size_t KohonenCudaPlanarImage2D::exampleSize() const {
	return m_stack.size();
}

KohonenOnMultispectralImages::KohonenOnMultispectralImages(KohonenImage2D* image, bool tmapOrPca, bool takeImageOwnership) {
	m_image = image;
	m_isOwnerOfImage = takeImageOwnership;
	m_tmapOrPca = tmapOrPca;
}

KohonenOnMultispectralImages::~KohonenOnMultispectralImages() {
	if (m_isOwnerOfImage) {
		delete m_image;
	}
}

bool KohonenOnMultispectralImages::setOutputHorizonProperties(RgbLayerFromDataset* saveProperties, QString tmapLabel) {
	bool result = false;
	if (saveProperties->width()==m_image->dimI() && saveProperties->depth()==m_image->dimJ()) {
		m_computationResultRgb = saveProperties;
		m_tmapLabel = tmapLabel;
		result = true;
	}
	return result;
}

bool KohonenOnMultispectralImages::setOutputHorizonProperties(FixedLayerFromDataset* saveProperties, QString tmapLabel) {
	bool result = false;
	if (saveProperties->width()==m_image->dimI() && saveProperties->depth()==m_image->dimJ()) {
		m_computationResultGray = saveProperties;
		m_tmapLabel = tmapLabel;
		result = true;
	}
	return result;
}

gsl_matrix* pca(const gsl_matrix* data, unsigned int L)
{
    /*
    @param data - matrix of data vectors, MxN matrix, each column is a data vector, M - dimension, N - data vector count
    @param L - dimension reduction
    */
    //assert(data != NULL);
    //assert(L > 0 && L < data->size2);
    unsigned int i;
    unsigned int rows = data->size1;
    unsigned int cols = data->size2;
    gsl_vector* mean = gsl_vector_alloc(rows);

    for(i = 0; i < rows; i++) {
        gsl_vector_set(mean, i, gsl_stats_mean(data->data + i * cols, 1, cols));
    }

    // Get mean-substracted data into matrix mean_substracted_data.
    gsl_matrix* mean_substracted_data = gsl_matrix_alloc(rows, cols);
    gsl_matrix_memcpy(mean_substracted_data, data);
    for(i = 0; i < cols; i++) {
        gsl_vector_view mean_substracted_point_view = gsl_matrix_column(mean_substracted_data, i);
        gsl_vector_sub(&mean_substracted_point_view.vector, mean);
    }
    gsl_vector_free(mean);

    // Compute Covariance matrix
    gsl_matrix* covariance_matrix = gsl_matrix_alloc(rows, rows);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0 / (double)(cols - 1), mean_substracted_data, mean_substracted_data, 0.0, covariance_matrix);
    gsl_matrix_free(mean_substracted_data);

    // Get eigenvectors, sort by eigenvalue.
    gsl_vector* eigenvalues = gsl_vector_alloc(rows);
    gsl_matrix* eigenvectors = gsl_matrix_alloc(rows, rows);
    gsl_eigen_symmv_workspace* workspace = gsl_eigen_symmv_alloc(rows);
    gsl_eigen_symmv(covariance_matrix, eigenvalues, eigenvectors, workspace);
    gsl_eigen_symmv_free(workspace);
    gsl_matrix_free(covariance_matrix);

    // Sort the eigenvectors
    gsl_eigen_symmv_sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_ABS_DESC);
    gsl_vector_free(eigenvalues);

    // Project the original dataset
    gsl_matrix* result = gsl_matrix_alloc(L, cols);
    gsl_matrix_view L_eigenvectors = gsl_matrix_submatrix(eigenvectors, 0, 0, rows, L);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &L_eigenvectors.matrix, data, 0.0, result);
    gsl_matrix_free(eigenvectors);

    // Result is n LxN matrix, each column is the original data vector with reduced dimension from M to L
    return result;
}


bool KohonenOnMultispectralImages::compute(int kohonenMapSize, int trainExamplesStep) {
	if (!m_tmapOrPca) {
		gsl_matrix* m = gsl_matrix_alloc(m_image->exampleSize(), m_image->dimI()*m_image->dimJ());
		gsl_matrix* res;
		std::vector<float> exemples ;
		exemples.resize(m_image->exampleSize()) ;
		std::size_t count = 0;
		for (std::size_t j=0; j<m_image->dimJ(); j++) {
			for (std::size_t i=0; i<m_image->dimI(); i++) {
				m_image->getExample(exemples.data(), i, j);
				for (std::size_t k=0; k<m_image->exampleSize(); k++) {
					gsl_matrix_set(m, k, count, exemples[k]);
				}
				count++;
			}
		}
		res = pca(m, m_image->exampleSize());
		gsl_matrix_free(m);

		std::vector<float> buffer;
		buffer.resize(m_image->dimI()*m_image->dimJ());
		for (std::size_t k=0; k<m_image->exampleSize(); k++) {
			count = 0;
			for (std::size_t j=0; j<m_image->dimJ(); j++) {
				for (std::size_t i=0; i<m_image->dimI(); i++) {
					if (k==0) {
						buffer[count] = gsl_matrix_get(res, k, count);
					} else {
						buffer[count] = gsl_matrix_get(res, 0, count) - gsl_matrix_get(res, k, count);
					}
					count++;
				}
			}
			if (m_computationResultRgb!=nullptr) {
				m_computationResultRgb->writeProperty(buffer.data(), m_tmapLabel + QString::number(k+1));
			}
		}

		gsl_matrix_free(res);
	} else {
		KohonenEngine kohonen(m_image->exampleSize(), kohonenMapSize);

		long DimInput = m_image->exampleSize();
		long ne_tailleCarte = kohonenMapSize ;
		int step = trainExamplesStep ;

		long dimy = m_image->dimI(); //m_dataset->width();
		long dimz = m_image->dimJ(); //m_dataset->depth();
		std::vector<float> label;
		label.resize(dimy*dimz) ;
		int ind_ex=0  ;
		int NbExempl = 10000 ;
		std::vector<float> exemples ;
		exemples.resize(NbExempl*DimInput) ;

		bool ok;

		for(int iz = 0 ; iz < dimz ; iz +=step ) {
			qDebug() << " plan numero " << iz+1 << "/ " << dimz;
			for(int iy = 0; iy < dimy ; iy +=step ) {

				if(m_image->isExampleValid(iy, iz)) {
					if(ind_ex >= NbExempl) {
						NbExempl += 1000 ;
						exemples.resize((long) NbExempl*DimInput) ;
					}
					m_image->getExample(exemples.data()+ind_ex*DimInput, iy, iz);
					ind_ex ++ ;
				}
			}
		}
		double  distance_E,distance_Q,distance, travail,w,log(),exp(),sqrt();
		double som, som2,som3;
		float   passigma, pasepsil, sigma, epsilon, epsilov;
		float   sigmamax = 1.0;
		float   sigmamin = 0.3;
		float   epsilmax = 0.7;
		float   epsilmin = 0.01;
		float   a0 = 5.2;
		float   sigma0;
		unsigned long   is = 8191;
		long    i, j, k, ij, ex, cy, compteur, tailltot, produit;
		long    nbdimcar = 1, nbcycles = 1000;
		long    dimcarte[5];
		std::vector<float> carteTop;
		long     voisinag = 1;
		/* Normalisation des exemples */
		long typenorm =3 , ne_nbCycles = 1000 ;
		long ne_tailleVoisin = 7 ;
		/* Creation de la carte topologique */

		tailltot = ne_tailleCarte * DimInput;
		carteTop.resize(tailltot);
		/* Normalisation des exemples */


		kohonen.trainTmap(exemples.data(),NbExempl,ne_nbCycles,
				ne_tailleVoisin,sigmamax,sigmamin,epsilmax,epsilmin,voisinag,typenorm);

		std::vector<float> example;
		example.resize(DimInput);

		for(int iz = 0 ; iz < dimz ; iz ++ ) {
			printf(" plan numero %d/ %d \n",iz + 1, dimz ) ;
			for(int iy = 0; iy < dimy ; iy ++ ) {
				if(m_image->isExampleValid(iy, iz)) {

					m_image->getExample(example.data(), iy, iz);

					bool ok;
					long val = kohonen.applyTmap(example.data(), &ok);
					if (ok) {
						label[iz*dimy+iy] = val;
					} else {
						label[iz*dimy+iy] = -1;
					}

				}
			}
		}

		if (m_computationResultGray!=nullptr) {
			m_computationResultGray->writeProperty(label.data(), m_tmapLabel);
		}
	}

	return true;
}
