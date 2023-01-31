/*
 * Layering.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "layering.h"

#include <iostream>
#include <fstream>
#include <cmath>

//#include "SampleType.h"
//#include "Cube.h"
#include "smtopo3ddesc.h"
#include <QByteArray>
#include <QRect>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "sismagedbmanager.h"
//#include "LayerProcess.h"
#include "layer.h"
#include "smsurvey3D.h"
//#include "util/RawUtil.h"
//#include "SampleTypeBinder.h"
#include "utils/stringutil.h"
#include "ioutil.h"
#include "LayerSlice.h"
#include "affinetransformation.h"
#include "seismic3ddataset.h"
#include "cubeseismicaddon.h"

namespace fs = boost::filesystem;


/**
 * "desc" file discribe Layering attributes. It was located in LAYER DESC DIRECTORY
 *
 * dirName is the directory of Layering inside one single survey3D
 */
Layering::Layering(const std::string& dirName) : m_dirName( dirName) {

	// dirName is in Layers directory of Survey3D
	// We read desc in DATA/LAYERS

	fs::path dp(dirName);
	const std::string folderPath = dp.parent_path().c_str();
	// Exemple: GT_Layering_dh1yMAeI._dh1yMXeM
	const std::string layeringName = dp.filename().c_str();

	size_t found = layeringName.find_last_of(".");
	const std::string base = layeringName.substr(0, found);

	fs::path descPath = folderPath;
	std::cout << descPath.c_str() << std::endl;
	descPath /= "/../../../../LAYERS/";
	std::cout << descPath.c_str() << std::endl;
	descPath /= base;
	std::cout << descPath.c_str() << std::endl;

	struct stat buffer;
	if ( access ( descPath.c_str(), F_OK) == -1)
		return;

	// DESC FILE in DATA/LAYERS

	for( const auto & entry : fs::directory_iterator(descPath)) {
		std::cout << entry.path() << std::endl;
		if ( endsWith(entry.path().c_str(), "desc") == 1) {
			readLayeringDesc(entry.path().c_str());
			break;
		}
	}
}

Layering::~Layering() {
	// TODO Auto-generated destructor stub
}

bool Layering::init() {
	if (m_layersInitialized)
		return true;

	// LAYER DIRECTORY

	for( const auto & entry : fs::directory_iterator(m_dirName)) {
		std::cout << entry.path() << std::endl;
		QString c( entry.path().c_str() );

		m_monoLayerPath = entry.path().c_str();
		m_monoLayer.setLayerDirectory(m_monoLayerPath);

		break;
	}
	m_layersInitialized = true;
	return true;
}

/**
 * Ecample:
#Thu Apr 02 12:43:36 CEST 2020
name=GT_Layering
nbLayer=1
type=Geotime
category=
 *
 */
bool  Layering::readLayeringDesc(const std::string& filename) {

	std::string line;
	std::ifstream myfile( filename.c_str() );
	if (myfile.is_open()) {
		while ( std::getline (myfile,line) ) {
			int len = line.length();
			const size_t b = line.find_last_of("=");
			if ( b >= len)
				continue;
			const std::string nameStr = line.substr(0, b);
			const std::string valueStr = (b+1 <= len-1) ? line.substr(b+1, len-1) : "";
			if ( nameStr.compare("name") == 0) {
				m_label = valueStr.c_str();
			}
			else if ( nameStr.compare("nbLayer") == 0) {
				m_nbLayer = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("type") == 0) {
				m_type = valueStr.c_str();
			}
			else if ( nameStr.compare("category") == 0) {
				m_category = valueStr.c_str();
			}
		}
		myfile.close();
	}
	else {
		return false;
		//throw io::CubeIOException(std::string("Unable to open file ") + filename.c_str());
	}
	myfile.close();
	return true;
}

template<typename InputType>
struct ReadSubData {
	static void run(LayerSlice* layer, const QRect& sourceRegion, int spectrumSlice, float* f3) {

		//const process::LayerProcess* spectrumDeconProcess = data->getLayerProcess();

		long width = sourceRegion.width();
		/*if (sourceRegion.right()==spectrumDeconProcess->getDimW() && spectrumDeconProcess->getDimW()%2==1) {
			width -= 1;
		}*/

		if (width>0) {
			const float* data = layer->getModuleData(spectrumSlice);

			for (std::size_t j=0; j<sourceRegion.height(); j++) {
				std::size_t index_data = (j+sourceRegion.top()) * layer->width() + sourceRegion.left();
				std::size_t index_buf = j * sourceRegion.width();

				memcpy(f3+index_buf, data+index_data, width * sizeof(float));
			}
		}
	}
};

bool Layering::writeProp(/*io::SampleTypeBinder& binder, */LayerSlice* data, QRect& sourceRegion,
		float* oriTab, float* floatTab, int slice, std::string& propName,
		double firstSvInline, double inlineSvDim, double inlineSvStep,
		double firstSvXline, double xlineSvDim, double xlineSvStep,
		double firstDsInline, double inlineDsDim, double inlineDsStep,
		double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey) {

	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	/*binder.bind <*/ ReadSubData<float>::run /*>*/ (data, sourceRegion, slice, oriTab );
	// Step factor is supposed to be and integer
	int xlineStepFact = xlineDsStep / xlineSvStep;
	int inlineStepFact = inlineDsStep / inlineSvStep;
	for ( int i = 0; i < inlineDsDim; i++) {
		for ( int x = 0; x < xlineDsDim; x++) {
			size_t dsIndex = i * xlineDsDim + x;
			size_t svIndex = ((firstDsInline - firstSvInline) / inlineSvStep + i * inlineStepFact) * xlineSvDim +
					((firstDsXline - firstSvXline) / xlineSvStep + x * xlineStepFact);
			floatTab[svIndex] = oriTab[dsIndex];
		}
	}
	return m_monoLayer.writeProperty(floatTab, propName);
}

bool Layering::writePropIsochron(/*io::SampleTypeBinder& binder, */LayerSlice* data, QRect& sourceRegion,
		float* oriTab, float* floatTab, int slice, std::string& propName,
		double firstSample, double sampleStep,
		double firstSvInline, double inlineSvDim, double inlineSvStep,
		double firstSvXline, double xlineSvDim, double xlineSvStep,
		double firstDsInline, double inlineDsDim, double inlineDsStep,
		double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey) {

	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	/*binder.bind <*/ ReadSubData<float>::run /*>*/ (data, sourceRegion, slice, oriTab );
	// Step factor is supposed to be and integer
	int xlineStepFact = xlineDsStep / xlineSvStep;
	int inlineStepFact = inlineDsStep / inlineSvStep;
	for ( int i = 0; i < inlineDsDim; i++) {
		for ( int x = 0; x < xlineDsDim; x++) {
			size_t dsIndex = i * xlineDsDim + x;
			size_t svIndex = ((firstDsInline - firstSvInline) / inlineSvStep + i * inlineStepFact) * xlineSvDim +
					((firstDsXline - firstSvXline) / xlineSvStep + x * xlineStepFact);
			floatTab[svIndex] = oriTab[dsIndex] * sampleStep + firstSample;
		}
	}
	return m_monoLayer.writeProperty(floatTab, propName);
}

/**
 * Save all processed maps on Sismage Layer
 *
 * TODO: Should be revisited when SeismicAddon will be completed
 */
void Layering::saveInto(LayerSlice* data) {
	if(!data)
		return;

	init();

	// Processed
	int dimWProcess = data->width();
	int dimHProcess = data->depth();
	size_t mapLengthProcess = dimWProcess * dimHProcess;

	QRect sourceRegion(0, 0, dimWProcess, dimHProcess);
	//quint32 resultLen = (quint32) (mapLength * sizeof(short));
	QByteArray shortBufferInDs;
	shortBufferInDs.reserve(mapLengthProcess * sizeof(float));
	float* floatTabFromProcess = static_cast<float*> (static_cast<void*>(shortBufferInDs.data()));

	//TODO swap bytes should be done here to save time

	// Layer Sismage
	int dimWSurvey = m_monoLayer.getNbTraces();
	int dimHSurvey = m_monoLayer.getNbProfiles();

	QByteArray floatBuffer;
	size_t mapLengthSurvey = dimWSurvey * dimHSurvey;
	floatBuffer.reserve(mapLengthSurvey * sizeof(float));
	float* floatTab = static_cast<float*> (static_cast<void*>(floatBuffer.data()));
	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	float zero = 0;
	switch_list_endianness_inplace(&zero, 4, 1);

	int nbOutputSlice = data->getNbOutputSlices();
	int method = data->getMethod();
	Seismic3DDataset* seismicS = data->seismic();
	float pasEch = seismicS->sampleTransformation()->a() / 1000;
	//

	std::string topo3dDescPath =
			SismageDBManager::getTopo3dDescfromLayeringPath(m_dirName.c_str());
	CubeSeismicAddon seismicAddon = seismicS->cubeSeismicAddon();

	SmTopo3dDesc topo3dDesc(topo3dDescPath);

	fs::path layeringPath(m_dirName);
	fs::path layersDirPath = layeringPath.parent_path();
	fs::path surveyDataPath = layersDirPath.parent_path();
	fs::path surveyPath = surveyDataPath.parent_path();
	SmSurvey3D survey3D(surveyPath.string());
	double firstSvInline = survey3D.firstInline();
	int inlineSvDim = survey3D.inlineDim();
	double inlineSvStep = survey3D.inlineStep();
	double firstSvXline = survey3D.firstXline();
	int xlineSvDim = survey3D.xlineDim();
	double xlineSvStep = survey3D.xlineStep();

	double firstDsInline = seismicAddon.getFirstInline();
	double inlineDsDim = dimHProcess;
	double inlineDsStep = seismicAddon.getInlineStep();
	double firstDsXline = seismicAddon.getFirstXline();
	double xlineDsDim = dimWProcess;
	double xlineDsStep =  seismicAddon.getXlineStep();
	//io::SampleTypeBinder binder(data->getLayerProcess()->getCubeS()->getNativeType());

	std::string propName = "Isochron";
	writePropIsochron(/*binder, */data, sourceRegion, floatTabFromProcess, floatTab, 0, propName,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);

	propName = "top";
	writePropIsochron(/*binder, */data, sourceRegion, floatTabFromProcess, floatTab, 0, propName,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);

	propName = "bottom";
	writePropIsochron(/*binder, */data, sourceRegion, floatTabFromProcess, floatTab, 0, propName,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);

	propName = "Amplitude";
	writeProp(/*binder, */data, sourceRegion, floatTabFromProcess, floatTab, 1, propName,
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);

	switch (method) {
		case 0: { // Morlet
			int freqMin = data->getFreqMin();
			int freqMax = data->getFreqMax();
			int freqStep = data->getFreqStep();
			for (int s = 1; s < nbOutputSlice - 1; s++) {
				int num = freqMin + (s - 1) * freqStep;
				char buf[12];
				sprintf(buf, "Frequency%03d", num);
				propName = buf;
				std::cout << "Export Morlet to Sismage Store: " << propName << std::endl;
				writeProp(/*binder,*/ data, sourceRegion, floatTabFromProcess, floatTab, s, propName,
						firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
						firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);
			}
			break;
		}
		case 1: { // Spectrum
			int windowSize = data->getWindowSize();
			double stepFreq = 1.0/(pasEch * (windowSize - 1));
			for (int s = 2; s < nbOutputSlice; s++) {
				int freq = (int) std::round(stepFreq * (s - 1));
				char buf[11];
				sprintf(buf, "Spectrum%03d", freq);
				propName = buf;
				std::cout << "Export Spectrum to Sismage Store: " << propName << std::endl;
				writeProp(/*binder,*/ data, sourceRegion, floatTabFromProcess, floatTab, s, propName,
						firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
						firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);
			}
			break;
		}
		case 2: { // GCC
			int windowSize = data->getWindowSize();
			int n = 2; // ?????????????????????????
			for (int s = 2; s < nbOutputSlice; s++) {
				char buf[11];
				sprintf(buf, "GCC%02d", s - 1);
				propName = buf;
				std::cout << "Export GCC to Sismage Store: " << propName << std::endl;
				writeProp(/*binder, */data, sourceRegion, floatTabFromProcess, floatTab, s, propName,
						firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
						firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey);
			}
			break;
		}
	}
}

