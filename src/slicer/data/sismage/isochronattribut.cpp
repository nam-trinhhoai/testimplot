/*
 * isochronattribut.cpp
 *
 *  Created on: Jan, 10, 2023
 *      Author: l0483271
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "isochronattribut.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>


#include "Xt.h"

#include <QByteArray>
#include <QRect>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "affinetransformation.h"
#include "cubeseismicaddon.h"
#include "interpolation.h"
#include "ioutil.h"
#include "layer.h"
#include "LayerSlice.h"
#include "seismic3ddataset.h"
#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "smtopo3ddesc.h"
#include "util_filesystem.h"
#include "utils/stringutil.h"

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
IsochronAttribut::IsochronAttribut(const Isochron& isochron, const std::string& attributName) :
		m_isochron(isochron),
		m_attributName(attributName) {

	m_attributPath = SismageDBManager::surveyPath2HorizonsPath(m_isochron.surveyPath());
	m_attributPath.append("/");
	m_attributPath.append(m_attributName);
	m_attributPath.append(".");
	m_attributPath.append(m_isochron.horizonName());
	m_attributPath.append(".amp");
}

IsochronAttribut::~IsochronAttribut() {
	// TODO Auto-generated destructor stub
}

bool IsochronAttribut::writePropIsochron(QRect& sourceRegion,
		short* attrTab, short* surveyTab, int slice, std::string& propName,
		SampleUnit sampleUnit,
		double firstSample, double sampleStep,
		double firstSvInline, double inlineSvDim, double inlineSvStep,
		double firstSvXline, double xlineSvDim, double xlineSvStep,
		double firstDsInline, double inlineDsDim, double inlineDsStep,
		double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey,
		bool interpolate) {

	for ( int i = 0; i < mapLengthSurvey; i++)
		surveyTab[i] = -9999;

	// Step factor is supposed to be and integer
	int xlineStepFact = xlineDsStep / xlineSvStep;
	int inlineStepFact = inlineDsStep / inlineSvStep;
	int valueMax = std::numeric_limits<short>::min();
	int valueMin = std::numeric_limits<short>::max();
	for ( int i = 0; i < inlineDsDim; i++) {
		for ( int x = 0; x < xlineDsDim; x++) {
			long inlineSvIndex = ((firstDsInline - firstSvInline) / inlineSvStep + i * inlineStepFact);
			long xlineSvIndex = ((firstDsXline - firstSvXline) / xlineSvStep + x * xlineStepFact);

			if (inlineSvIndex>=0 && inlineSvIndex<inlineSvDim && xlineSvIndex>=0 && xlineSvIndex<xlineSvDim) {
				size_t dsIndex = i * xlineDsDim + x;
				size_t svIndex = inlineSvIndex * xlineSvDim + xlineSvIndex;
				surveyTab[svIndex] = attrTab[dsIndex] /** sampleStep + firstSample*/;
				if ( surveyTab[svIndex] > valueMax)
					valueMax = surveyTab[svIndex];
				if ( surveyTab[svIndex] < valueMin)
					valueMin = surveyTab[svIndex];
			}
		}
	}

	if (interpolate && (inlineStepFact>1 || xlineStepFact>1)) {
		Isochron::interpolateBuffer(surveyTab, inlineDsDim, inlineSvDim, xlineDsDim, xlineSvDim,
				firstDsInline, firstSvInline, inlineSvStep, inlineStepFact, firstDsXline, firstSvXline,
				xlineSvStep, xlineStepFact, -9999.0);
	}

	std::cout << "Attribute file: " <<m_attributPath << std::endl;
	std::cout << "Value Max: " << valueMax << " Value Min: " << valueMin << std::endl;

	std::string geometryDesc = m_isochron.surveyPath();
	geometryDesc.append("/DATA/SEISMIC/geometry.desc");

	std::cout << "GemetryDesc file: " << geometryDesc << std::endl;
	std::cout << "Attribute file: " << m_attributPath << std::endl;

	inri::Xt* geometryDescXt = new inri::Xt(geometryDesc);
	if (!geometryDescXt->is_valid())
		std::cout <<"xt cube is not valid" << std::endl;

	std::cout << "DIMX= " << geometryDescXt->nRecords() << " DIMY= " << geometryDescXt->nSlices() <<
			std::endl;

	// Should be rewrited perhaps in Xt
	{
		inri::Xt* isochronXt = new inri::Xt(m_attributPath,
			geometryDescXt->nRecords(), geometryDescXt->nSlices(), 1,
			inri::Xt::Type::Signed_16, 1);
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
		isochronXt->writeSlice(0, surveyTab);
	}

	//isochronXt->copy_comments(*geometryDescXt);
    struct stat buffer;
//    if (stat(fn.c_str(), &buffer) == 0
//        && buffer.st_size>0) { // file exists && file not empty
	struct nf_fmt ifmt;
	struct image*  im = image_(const_cast<char*>(m_attributPath.c_str()), "s", " ", &ifmt);
	int xx = iputhline(im, "Kind=", "Amplitude");
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

void IsochronAttribut::saveInto(CPUImagePaletteHolder *cpuIso, const CubeSeismicAddon& seismicAddon, bool interpolate) {
	if(!cpuIso)
		return;

	QPair<bool, QStringList> res = mkpath(QFileInfo(QString::fromStdString(m_attributPath)).absolutePath());
	if (!res.first) {
		return;
	}

	// Processed
	int dimWProcess = cpuIso->width();
	int dimHProcess = cpuIso->height();
	size_t mapLengthProcess = dimWProcess * dimHProcess;

	QRect sourceRegion(0, 0, dimWProcess, dimHProcess);

	short* attrBuf = static_cast<short*>(cpuIso->backingPointer());
	QVector2D isoRange = cpuIso->range();

	// Survey
	SmSurvey3D survey3D(m_isochron.surveyPath());
	std::string topo3dDescPath =
			SismageDBManager::getTopo3dDescfromSurveyPath(m_isochron.surveyPath());

	SmTopo3dDesc topo3dDesc(topo3dDescPath);

	SampleUnit sampleUnit = seismicAddon.getSampleUnit();

	int dimWSurvey = survey3D.xlineDim(); //xx.getNbTraces();
	int dimHSurvey = survey3D.inlineDim(); //xx.getNbProfiles();

	QByteArray shortBuffer;
	size_t mapLengthSurvey = dimWSurvey * dimHSurvey;
	shortBuffer.reserve(mapLengthSurvey * sizeof(short));
	short* surveyTab = static_cast<short*> (static_cast<void*>(shortBuffer.data()));
	for ( int i = 0; i < mapLengthSurvey; i++)
		surveyTab[i] = -9999;

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

	std::string propName = m_attributName;
	writePropIsochron(sourceRegion, attrBuf, surveyTab, 0, propName, sampleUnit,
			seismicAddon.getFirstSample(), seismicAddon.getSampleStep(),
			firstSvInline, inlineSvDim, inlineSvStep, firstSvXline, xlineSvDim, xlineSvStep,
			firstDsInline, inlineDsDim, inlineDsStep, firstDsXline, xlineDsDim, xlineDsStep, mapLengthSurvey, interpolate);
}
