#include "KohonenProcess.h"
#include "seismic3ddataset.h"

#include "fixedlayerfromdataset.h"
#include "iabstractisochrone.h"
#include "KohonenEngine.h"
#include "sampletypebinder.h"

#include <vector>
#include <cmath>

#include <QDebug>


AbstractKohonenProcess::AbstractKohonenProcess(Seismic3DDataset* dataset, int channel) {
	m_dataset = dataset;
	m_channel = channel;
}

AbstractKohonenProcess::~AbstractKohonenProcess() {
	if (m_haveResultOwnerShip && m_computationResult!=nullptr) {
		delete m_computationResult;
	}
}

AbstractKohonenProcess* AbstractKohonenProcess::getObjectFromDataset(Seismic3DDataset* dataset, int channel) {
	// only short for now
	AbstractKohonenProcess* out = nullptr;
	SampleTypeBinder binder(dataset->sampleType());
	out = binder.bind<CreateKohonenProcessObjectKernel>(dataset, channel);
	//KohonenProcess<short>* out = new KohonenProcess<short>(dataset, channel);
	return out;
}

bool AbstractKohonenProcess::setExtractionIsochrone(IAbstractIsochrone* isochrone) {
	bool result = false;
	if (isochrone!=nullptr && isochrone->getNumTraces()==m_dimI && isochrone->getNumProfils()==m_dimJ) {
		m_isochrone = isochrone;
		result = true;
	}
	return result;
}

bool AbstractKohonenProcess::setOutputHorizonProperties(FixedLayerFromDataset* saveProperties, QString tmapLabel) {
	bool result = false;
	if (saveProperties->width()==m_dimI && saveProperties->depth()==m_dimJ) {
		releaseComputationResult();
		m_haveResultOwnerShip = false;
		m_computationResult = saveProperties;
		m_tmapLabel = tmapLabel;
		result = true;
	}
	return result;
}

bool AbstractKohonenProcess::setOutputHorizonBuffer(float* tab) {
	m_outputHorizonBuffer = tab;
	return true;
}

bool AbstractKohonenProcess::setOutputOnIsochroneAttribute(float* tab) {
	m_outputOnIsochroneAttribute = tab;
	return true;
}

// Return current result holder, it can be nullptr
FixedLayerFromDataset* AbstractKohonenProcess::getComputationResult() {
	return m_computationResult;
}

QString AbstractKohonenProcess::tmapLabel() const {
	return m_tmapLabel;
}

FixedLayerFromDataset* AbstractKohonenProcess::releaseComputationResult() {
	m_haveResultOwnerShip = false;
	FixedLayerFromDataset* tmp = m_computationResult;
	m_computationResult = nullptr;
	return tmp;
}

bool AbstractKohonenProcess::hasResultOwnership() const {
	return m_haveResultOwnerShip;
}

bool AbstractKohonenProcess::isNonValue(float testVal) const {
	return testVal == m_nonValue;
}

long AbstractKohonenProcess::dimI() const {
	return m_dimI;
}

long AbstractKohonenProcess::dimJ() const {
	return m_dimJ;
}

bool AbstractKohonenProcess::compute(int exampleSize, int kohonenMapSize, int trainExamplesStep) {
	if (m_computationResult==nullptr && m_outputHorizonBuffer==nullptr) {
		m_haveResultOwnerShip = true;
		m_computationResult = new FixedLayerFromDataset("Tmap", nullptr, m_dataset);
	}

	KohonenEngine kohonen(exampleSize, kohonenMapSize);

	long DimInput = exampleSize ;
	long ne_tailleCarte = kohonenMapSize ;
	int step = trainExamplesStep ;

	long dimy = m_dimI; //m_dataset->width();
	long dimz = m_dimJ; //m_dataset->depth();
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

			if(isExampleValid(DimInput, iy, iz)) {
				if(ind_ex >= NbExempl) {
					NbExempl += 1000 ;
					exemples.resize((long) NbExempl*DimInput) ;
				}
				getExample(exemples.data()+ind_ex*DimInput, DimInput, iy, iz);
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

	float tdeb = m_dataset->sampleTransformation()->b();
	float pasech = m_dataset->sampleTransformation()->a();

	for(int iz = 0 ; iz < dimz ; iz ++ ) {
		printf(" plan numero %d/ %d \n",iz + 1, dimz ) ;
		for(int iy = 0; iy < dimy ; iy ++ ) {
			if(isExampleValid(DimInput, iy, iz)) {

				getExample(example.data(), DimInput, iy, iz);

				if (m_outputOnIsochroneAttribute!=nullptr) {
					m_outputOnIsochroneAttribute[iz*dimy+iy] = example[0];
				}

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

	if (m_computationResult!=nullptr) {
		m_computationResult->writeProperty(label.data(), m_tmapLabel);
	}
	if (m_outputHorizonBuffer!=nullptr) {
		memcpy(m_outputHorizonBuffer, label.data(), label.size()*sizeof(float));
	}

	return true;
}
