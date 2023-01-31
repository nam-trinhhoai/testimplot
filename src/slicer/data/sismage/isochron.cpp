/*
 * Isochron.cpp
 *
 *  Created on: Dec, 8, 2020
 *      Author: Georges
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "isochron.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>


#include "Xt.h"

#include "smtopo3ddesc.h"
#include <QByteArray>
#include <QRect>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "sismagedbmanager.h"
#include "layer.h"
#include "smsurvey3D.h"
#include "utils/stringutil.h"
#include "ioutil.h"
#include "LayerSlice.h"
#include "affinetransformation.h"
#include "seismic3ddataset.h"
#include "cubeseismicaddon.h"
#include "sismagedbmanager.h"
#include "util_filesystem.h"

namespace fs = boost::filesystem;


extern "C" {
#define Linux 1
#include "image.h"
#include "comOCR.h"
int iputhline(struct image *nf, char *key, char *buf);
}

/**
 * Create a neew isochron
 */
Isochron::Isochron(const std::string& horizonName, const std::string& surveyPath) :
		m_surveyPath(surveyPath),
		m_horizonName(horizonName){

	m_isochronPath = SismageDBManager::surveyPath2HorizonsPath(m_surveyPath);
	m_isochronPath.append("/");
	m_isochronPath.append(horizonName);
	m_isochronPath.append(".iso");
}

Isochron::~Isochron() {
	// TODO Auto-generated destructor stub
}

std::string Isochron::horizonName() const {
	return m_horizonName;
}

std::string Isochron::surveyPath() const {
	return m_surveyPath;
}

template<typename InputType>
struct ReadSubData {
	static void run(LayerSlice* layer, const QRect& sourceRegion, int spectrumSlice, short* f3) {

		//const process::LayerProcess* spectrumDeconProcess = data->getLayerProcess();

		long width = sourceRegion.width();
		/*if (sourceRegion.right()==spectrumDeconProcess->getDimW() && spectrumDeconProcess->getDimW()%2==1) {
			width -= 1;
		}*/

		if (width>0) {
			const short* data = layer->getModuleData(spectrumSlice);

			for (std::size_t j=0; j<sourceRegion.height(); j++) {
				std::size_t index_data = (j+sourceRegion.top()) * layer->width() + sourceRegion.left();
				std::size_t index_buf = j * sourceRegion.width();

				memcpy(f3+index_buf, data+index_data, width * sizeof(short));
			}
		}
	}
};

bool Isochron::writePropIsochron(QRect& sourceRegion,
		short* shortTab, float* floatTab, int slice, std::string& propName,
		SampleUnit sampleUnit,
		double firstSample, double sampleStep,
		double firstSvInline, double inlineSvDim, double inlineSvStep,
		double firstSvXline, double xlineSvDim, double xlineSvStep,
		double firstDsInline, double inlineDsDim, double inlineDsStep,
		double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey,
		bool interpolate) {

	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	// Step factor is supposed to be and integer
	int xlineStepFact = xlineDsStep / xlineSvStep;
	int inlineStepFact = inlineDsStep / inlineSvStep;
	int valueMax = 0;
	int valueMin = 100000;
	for ( int i = 0; i < inlineDsDim; i++) {
		for ( int x = 0; x < xlineDsDim; x++) {
			long inlineSvIndex = ((firstDsInline - firstSvInline) / inlineSvStep + i * inlineStepFact);
			long xlineSvIndex = ((firstDsXline - firstSvXline) / xlineSvStep + x * xlineStepFact);

			if (inlineSvIndex>=0 && inlineSvIndex<inlineSvDim && xlineSvIndex>=0 && xlineSvIndex<xlineSvDim) {
				size_t dsIndex = i * xlineDsDim + x;
				size_t svIndex = inlineSvIndex * xlineSvDim + xlineSvIndex;
				floatTab[svIndex] = shortTab[dsIndex] /** sampleStep + firstSample*/;
				if ( floatTab[svIndex] > valueMax)
					valueMax = floatTab[svIndex];
				if ( floatTab[svIndex] < valueMin)
					valueMin = floatTab[svIndex];
			}
		}
	}

	if (interpolate && (inlineStepFact>1 || xlineStepFact>1)) {
		interpolateBuffer(floatTab, inlineDsDim, inlineSvDim, xlineDsDim, xlineSvDim,
				firstDsInline, firstSvInline, inlineSvStep, inlineStepFact, firstDsXline, firstSvXline,
				xlineSvStep, xlineStepFact, -9999.0);
	}

	std::cout << "Iso file: " <<m_isochronPath << std::endl;
	std::cout << "Value Max: " << valueMax << " Value Min: " << valueMin << std::endl;

	std::string geometryDesc = m_surveyPath;
	geometryDesc.append("/DATA/SEISMIC/geometry.desc");

	std::cout << "GemetryDesc file: " << geometryDesc << std::endl;
	std::cout << "Iso file: " << m_isochronPath << std::endl;

	inri::Xt* geometryDescXt = new inri::Xt(geometryDesc);
	if (!geometryDescXt->is_valid())
		std::cout <<"xt cube is not valid" << std::endl;

	std::cout << "DIMX= " << geometryDescXt->nRecords() << " DIMY= " << geometryDescXt->nSlices() <<
			std::endl;

	// Should be rewrited perhaps in Xt
	{
		inri::Xt* isochronXt = new inri::Xt(m_isochronPath,
			geometryDescXt->nRecords(), geometryDescXt->nSlices(), 1,
			inri::Xt::Type::Float, 1);
		if (!isochronXt->is_valid())
			std::cout <<"xt cube is not valid" << std::endl;

		float i_start = geometryDescXt->startSamples();
		float i_step = geometryDescXt->startRecord();
		float j_start = geometryDescXt->startSlice();
		float j_step = geometryDescXt->stepSamples();
		float k_start = geometryDescXt->stepRecords();
		float k_step = geometryDescXt->stepSlices();
		float interRecords = geometryDescXt->interRecords();
		float interSlice = geometryDescXt->interSlices();
		//Xt::Axis axis = Xt::Axis::Depth;

		isochronXt->insertComments(i_start, i_step,
		   j_start, j_step,
		   k_start, k_step,
		   interRecords, interSlice/*, axis*/);
		isochronXt->writeSlice(0, floatTab);
	}

	//isochronXt->copy_comments(*geometryDescXt);
    struct stat buffer;
//    if (stat(fn.c_str(), &buffer) == 0
//        && buffer.st_size>0) { // file exists && file not empty
	struct nf_fmt ifmt;
	struct image*  im = image_(const_cast<char*>(m_isochronPath.c_str()), "s", " ", &ifmt);
	int xx = iputhline(im, "Kind=", "Depth");
	QString dataExtentStr("0 0 ");
	dataExtentStr = dataExtentStr + QString::number(geometryDescXt->nSlices()) + " " + QString::number(geometryDescXt->nRecords());
	std::vector<char> dataExtentBuf;
	dataExtentBuf.resize(dataExtentStr.count()+1);
	dataExtentBuf[dataExtentStr.count()] = 0;
	memcpy(dataExtentBuf.data(), dataExtentStr.toStdString().c_str(), dataExtentStr.count());
	xx = iputhline(im, "DataExtent=", dataExtentBuf.data());
	//xx = iputhline(im, "DataExtent=", "0 0 700 1500"); // data extent example

	QString minStr = QString::number(valueMin);
	std::vector<char> valueMinBuf;
	valueMinBuf.resize(minStr.count()+1);
	valueMinBuf[minStr.count()] = 0;
	memcpy(valueMinBuf.data(), minStr.toStdString().c_str(), minStr.count());
	xx = iputhline(im, "VMIN=", valueMinBuf.data());

	QString maxStr = QString::number(valueMax);
	std::vector<char> valueMaxBuf;
	valueMaxBuf.resize(maxStr.count()+1);
	valueMaxBuf[maxStr.count()] = 0;
	memcpy(valueMaxBuf.data(), maxStr.toStdString().c_str(), maxStr.count());
	xx = iputhline(im, "VMAX=", valueMaxBuf.data());

	if (sampleUnit ==SampleUnit::DEPTH) {
		xx = irephline(im, "TYPE_AXE1=", 1, "3");
		xx = irephline(im, "TYPE_AXE2=", 1, "4");
		xx = irephline(im, "TYPE_AXE3=", 1, "2");
	} else {
		xx = irephline(im, "TYPE_AXE1=", 1, "2");
		xx = irephline(im, "TYPE_AXE2=", 1, "3");
		xx = irephline(im, "TYPE_AXE3=", 1, "4");
	}
	c_fermnf(im);
        //    } else {
//        im = NULL;
//    }
	return true;
}

/**
 * Save all processed maps on Sismage Layer
 *
 * TODO: Should be revisited when SeismicAddon will be completed
 */
void Isochron::saveInto(CUDAImagePaletteHolder *cudaIso, const CubeSeismicAddon& seismicAddon, bool interpolate) {
	if(!cudaIso)
		return;

	QPair<bool, QStringList> res = mkpath(QFileInfo(QString::fromStdString(m_isochronPath)).absolutePath());
	if (!res.first) {
		return;
	}

	// Processed
	int dimWProcess = cudaIso->width();
	int dimHProcess = cudaIso->height();
	size_t mapLengthProcess = dimWProcess * dimHProcess;

	QRect sourceRegion(0, 0, dimWProcess, dimHProcess);
	/*QByteArray shortBufferInDs;
	shortBufferInDs.reserve(mapLengthProcess * sizeof(short));*/
	//
	short* isoBuf = static_cast<short*>(cudaIso->backingPointer());
	QVector2D isoRange = cudaIso->range();

	/*for (int h = 0; h < dimHProcess; h++) {
		for (int w = 0; w < dimWProcess; w++) {
			int iIn = (h * dimWProcess + w);
			int iOut = (w * dimHProcess + h);
			shortBufferInDs[iOut]  = isoBuf[iIn];
		}
	}*/

	//TODO swap bytes should be done here to save time

	// Survey
	SmSurvey3D survey3D(m_surveyPath);
	std::string topo3dDescPath =
			SismageDBManager::getTopo3dDescfromSurveyPath(m_surveyPath);

	SmTopo3dDesc topo3dDesc(topo3dDescPath);

	SampleUnit sampleUnit = seismicAddon.getSampleUnit();

	int dimWSurvey = survey3D.xlineDim(); //xx.getNbTraces();
	int dimHSurvey = survey3D.inlineDim(); //xx.getNbProfiles();

	QByteArray floatBuffer;
	size_t mapLengthSurvey = dimWSurvey * dimHSurvey;
	floatBuffer.reserve(mapLengthSurvey * sizeof(float));
	float* floatTab = static_cast<float*> (static_cast<void*>(floatBuffer.data()));
	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	float zero = 0;
	switch_list_endianness_inplace(&zero, 4, 1);

	/* ?????????????
	Seismic3DDataset* seismicS = data->seismic();
	float pasEch = seismicS->sampleTransformation()->a() / 1000;
	*/

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
	writePropIsochron(sourceRegion, isoBuf, floatTab, 0, propName, sampleUnit,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey, interpolate);

}

void Isochron::saveInto(CPUImagePaletteHolder *cpuIso, const CubeSeismicAddon& seismicAddon, bool interpolate) {
	if(!cpuIso)
		return;

	QPair<bool, QStringList> res = mkpath(QFileInfo(QString::fromStdString(m_isochronPath)).absolutePath());
	if (!res.first) {
		return;
	}

	// Processed
	int dimWProcess = cpuIso->width();
	int dimHProcess = cpuIso->height();
	size_t mapLengthProcess = dimWProcess * dimHProcess;

	QRect sourceRegion(0, 0, dimWProcess, dimHProcess);

	short* isoBuf = static_cast<short*>(cpuIso->backingPointer());
	QVector2D isoRange = cpuIso->range();

	// Survey
	SmSurvey3D survey3D(m_surveyPath);
	std::string topo3dDescPath =
			SismageDBManager::getTopo3dDescfromSurveyPath(m_surveyPath);

	SmTopo3dDesc topo3dDesc(topo3dDescPath);

	SampleUnit sampleUnit = seismicAddon.getSampleUnit();

	int dimWSurvey = survey3D.xlineDim(); //xx.getNbTraces();
	int dimHSurvey = survey3D.inlineDim(); //xx.getNbProfiles();

	QByteArray floatBuffer;
	size_t mapLengthSurvey = dimWSurvey * dimHSurvey;
	floatBuffer.reserve(mapLengthSurvey * sizeof(float));
	float* floatTab = static_cast<float*> (static_cast<void*>(floatBuffer.data()));
	for ( int i = 0; i < mapLengthSurvey; i++)
		floatTab[i] = -9999.0;

	float zero = 0;
	switch_list_endianness_inplace(&zero, 4, 1);

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

	std::string propName = "Isochron";
	writePropIsochron(sourceRegion, isoBuf, floatTab, 0, propName, sampleUnit,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey, interpolate);
}

// inlineStepFact inlineDsStep / inlineSvStep
std::pair<double, double> Isochron::getStepFact(const std::string& surveyPath, CubeSeismicAddon seismicAddon) {
	SmSurvey3D survey3D(surveyPath);
	std::string topo3dDescPath =
			SismageDBManager::getTopo3dDescfromSurveyPath(surveyPath);

	SmTopo3dDesc topo3dDesc(topo3dDescPath);

	double inlineSvStep = survey3D.inlineStep();
	double xlineSvStep = survey3D.xlineStep();

	double inlineDsStep = seismicAddon.getInlineStep();
	double xlineDsStep =  seismicAddon.getXlineStep();

	double inlineStepFact = inlineDsStep / inlineSvStep;
	double xlineStepFact = xlineDsStep / xlineSvStep;

	return std::pair<double, double>(inlineStepFact, xlineStepFact);
}

bool Isochron::isXtHorizon(const std::string& horizonPath) {
	std::string cmd = "TestXtFile " + horizonPath + " > /dev/null 2>&1";
	int res = std::system(cmd.c_str());

	return res==0;
}

int Isochron::readIntFromHeader(FILE* pFile, const char* filter, std::size_t header_size, bool* ok) {
	fseek(pFile, 0x4c, SEEK_SET);
	char str[header_size];

	int n = 0, cont = 1;
	int value = -1;
	bool valid = false;
	while (cont) {
		int nbreType = fscanf(pFile, filter, &value);
		if (nbreType > 0) {
			cont = 0;
			valid = true;
		} else {
			fgets(str, header_size, pFile);
		}

		std::size_t pos = ftell(pFile);
		if (pos>=header_size) {
			cont = 0;
			strcpy(str, "Other");
		}
	}

	if (ok) {
		*ok = valid;
	}
	return value;
}

std::string Isochron::readStrFromHeader(FILE* pFile, const char* filter, std::size_t header_size, bool* ok) {
	fseek(pFile, 0x4c, SEEK_SET);
	char str[header_size];

	int n = 0, cont = 1;
	char out[header_size];
	bool valid = false;
	while (cont) {
		int nbreType = fscanf(pFile, filter, &out);
		if (nbreType > 0) {
			cont = 0;
			valid = true;
		} else {
			fgets(str, header_size, pFile);
		}

		std::size_t pos = ftell(pFile);
		if (pos>=header_size) {
			cont = 0;
			strcpy(str, "Other");
		}
	}

	if (ok) {
		*ok = valid;
	}
	return std::string(out);
}

SampleUnit Isochron::getSampleUnit(const std::string& horizonPath) {
	std::size_t offset;
	{
		inri::Xt xt(horizonPath.c_str());
		if (!xt.is_valid()) {
			return SampleUnit::NONE;
		}
		offset = (size_t)xt.header_size();
	}

	FILE *pFile = fopen(horizonPath.c_str(), "r");
	if (pFile == NULL) {
		return SampleUnit::NONE;
	}

	fseek(pFile, 0, SEEK_END);
	std::size_t size = ftell(pFile);
	if (size < offset) {
		fclose(pFile);
		return SampleUnit::NONE;
	}

	// get comments
	bool okAxe3;
	int typeAxe3 = readIntFromHeader(pFile, "TYPE_AXE3=\t%d\n", offset, &okAxe3);
	bool okNaturDon;
	std::string naturDon = readStrFromHeader(pFile, "NATUR_DON=\t%s\n", offset, &okNaturDon);

	fclose(pFile);

	// decide output type from comments
	SampleUnit fileKind = SampleUnit::NONE;
	if (okNaturDon && naturDon=="Time") {
		fileKind = SampleUnit::TIME;
	}
	if (okNaturDon && naturDon=="Depth") {
		fileKind = SampleUnit::DEPTH;
	}

	SampleUnit output = SampleUnit::NONE;
	if (okAxe3 && typeAxe3==2) {
		output = SampleUnit::DEPTH;
	} else if (okNaturDon && fileKind!=SampleUnit::DEPTH && fileKind!=SampleUnit::TIME) {
		output = SampleUnit::NONE;
	} else {
		output = SampleUnit::TIME;
	}

	return output;
}
