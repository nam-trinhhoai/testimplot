#include "BNNIJsonGenerator.h"
#include "wellhead.h"
#include "smdataset3D.h"
#include "smsurvey3D.h"
#include "sismagedbmanager.h"
#include "GeotimeProjectManagerWidget.h"
#include "Xt.h"
#include "imageformats.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "sampletypebinder.h"
#include "interpolation.h"
#include "cnxPatchIndex.h"
#include "datasetrelatedstorageimpl.h"
#include "nextvisiondbmanager.h"
#include "mtlengthunit.h"

#include <QProcess>
#include <QStringList>
#include <QDir>
#include <QFileInfo>
#include <QSaveFile>
#include <QDebug>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <gdal.h>

#include <boost/filesystem.hpp>

#include <rapidjson/error/en.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>

namespace fs = boost::filesystem;


#define BNNI_EPSILON 1.0e-30

BnniJsonGenerator::BnniJsonGenerator() {
	m_depthUnit = &MtLengthUnit::METRE;
}

BnniJsonGenerator::~BnniJsonGenerator() {
	clearWellBores();
}

void BnniJsonGenerator::clearWellBores() {
	for (WellBore* bore : m_wellBores) {
		bore->wellHead()->removeWellBore(bore);
		bore->deleteLater();
	}
	for (WellHead* head : m_wellHeads) {
		head->deleteLater();
	}
}

bool compareArray(const std::array<double,6>& array1, const std::array<double,6>& array2) {
	bool out = array1.size()==array2.size();
	std::size_t i = 0;
	while (out && i<array1.size()) {
		out = std::fabs(array1[i]-array2[i])<BNNI_EPSILON;
		i++;
	}
	return out;
}

bool BnniJsonGenerator::addInputVolume(const QString& path, const std::pair<float, float>& dynamic) {
	bool ok = false;

	QProcess process;
	QStringList options;
	options << path;
	process.start("TestXtFile", options);
	process.waitForFinished();

	if (process.exitCode()!=QProcess::NormalExit) {
		std::cerr << "provided file is not in xt format (" << path.toStdString() << ")" << std::endl;
	} else {
		SmDataset3D d3d(path.toStdString());
		inri::Xt xt(path.toStdString());
		AffineTransformation sampleTransfo = d3d.sampleTransfo();
		Affine2DTransformation inlineXlineTransfoForInline = d3d.inlineXlineTransfoForInline();
		Affine2DTransformation inlineXlineTransfoForXline = d3d.inlineXlineTransfoForXline();

//		params.sampleTransformation = &sampleTransfo;
//		params.ijToInlineXlineTransfoForInline = &inlineXlineTransfoForInline;
//		params.ijToInlineXlineTransfoForXline = &inlineXlineTransfoForXline;
		SmSurvey3D smSurvey(SismageDBManager::survey3DPathFromDatasetPath(path.toStdString()));
		std::array<double, 6> inlineXlineTransfo =
				smSurvey.inlineXlineToXYTransfo().direct();
		std::array<double, 6> ijToInlineXline = d3d.inlineXlineTransfo().direct();

		std::array<double, 6> res;
		GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
				res.data());

		Affine2DTransformation ijToXYTransfo(d3d.inlineXlineTransfo().width(),
				d3d.inlineXlineTransfo().height(), res);

		int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(path);
		SampleUnit unit = (timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH;

//		params.ijToXYTransfo = &ijToXYTransfo;

		if (m_paths.size()==0) {
			m_sampleTransform.reset(new AffineTransformation(sampleTransfo));
			m_inlineXlineTransfoForInline.reset(new Affine2DTransformation(inlineXlineTransfoForInline));
			m_inlineXlineTransfoForXline.reset(new Affine2DTransformation(inlineXlineTransfoForXline));
			m_ijToXYTransfo.reset(new Affine2DTransformation(ijToXYTransfo));
			m_seismicUnit = unit;
			m_numSamples = xt.nSamples();
			m_numTraces = xt.nRecords();
			m_numProfils = xt.nSlices();
			m_startTrace = xt.startRecord();
			m_stepTraces = xt.stepRecords();
			m_startProfil = xt.startSlice();
			m_stepProfils = xt.stepSlices();


			m_sampleTransformSurrechantillon.reset(new AffineTransformation(m_pasSampleSurrechantillon, xt.startSamples()));
			m_numSamplesSurrechantillon = std::floor((m_numSamples - 1) * xt.stepSamples() / m_pasSampleSurrechantillon + 1);
			ok = true;
		} else {
			ok = m_seismicUnit==unit && std::fabs(m_sampleTransform->a()-sampleTransfo.a())<BNNI_EPSILON &&
					std::fabs(m_sampleTransform->b()-sampleTransfo.b())<BNNI_EPSILON &&
					compareArray(m_inlineXlineTransfoForInline->direct(), inlineXlineTransfoForInline.direct()) &&
					compareArray(m_inlineXlineTransfoForXline->direct(), inlineXlineTransfoForXline.direct()) &&
					compareArray(m_ijToXYTransfo->direct(), ijToXYTransfo.direct()) && xt.nSamples()==m_numSamples &&
					xt.nRecords()==m_numTraces && xt.nSlices()==m_numProfils;
		}
		if (ok) {
			m_paths.push_back(path);
			m_seismicDynamics.push_back(dynamic);
			m_seismicWeights.push_back(1.0f);
		}
	}
	return ok;
}

bool BnniJsonGenerator::addWellBore(const QString& descPath, const QString& deviationPath,
			const QString tfpPath, const QString& tfpName, const std::vector<QString>& logPaths,
			const std::vector<QString>& logNames, const QString& wellHeadDescPath) {
	if (logNames.size()==0 || logNames.size()!=m_wellHeaders.size()) {
		return false;
	}

	bool ok = false;
	WellHead* wellHead = nullptr;
	bool wellHeadExists = false;

	std::size_t idxWellHead = 0;
	while (!wellHeadExists && idxWellHead<m_wellHeads.size()) {
		wellHeadExists = m_wellHeads[idxWellHead]->isIdPathIdentical(wellHeadDescPath);
		if (wellHeadExists) {
			wellHead = m_wellHeads[idxWellHead];
		} else {
			idxWellHead++;
		}
	}

	if (!wellHeadExists) {
		wellHead = WellHead::getWellHeadFromDescFile(wellHeadDescPath, nullptr);
	}
	if (wellHead!=nullptr) {
		std::vector<QString> tfpPaths, tfpNames;
		tfpPaths.push_back(tfpPath);
		tfpNames.push_back(tfpName);
		WellBore* wellBore = new WellBore(nullptr, descPath, deviationPath,
				tfpPaths, tfpNames, logPaths, logNames, wellHead);

		if (wellBore==nullptr) {
			if (!wellHeadExists) {
				delete wellHead;
			}
		} else {
			if (!wellHeadExists) {
				m_wellHeads.push_back(wellHead);
			}
			wellHead->addWellBore(wellBore);
			m_wellBores.push_back(wellBore);
			ok = true;
		}
	}
	return ok;
}

bool BnniJsonGenerator::addHorizonsInterval(const QString& topPath, float topDelta, const QString& bottomPath, float bottomDelta) {
	bool intervalValid = true;

	// use of && to avoid copies
	std::pair<std::vector<float>, std::vector<float>>&& intervalBuffers = std::pair<std::vector<float>, std::vector<float>>();
	intervalValid = readHorizon(topPath, intervalBuffers.first);

	if (intervalValid) {
		applyDelta(intervalBuffers.first, topDelta);
		intervalValid = readHorizon(bottomPath, intervalBuffers.second);
		intervalValid = intervalValid && intervalBuffers.first.size()==intervalBuffers.second.size();
	}

	if (intervalValid) {
		applyDelta(intervalBuffers.second, bottomDelta);

		std::size_t N = intervalBuffers.first.size();
		for (std::size_t idx = 0; idx<N; idx++) {
			float minVal = std::min(intervalBuffers.second[idx], intervalBuffers.first[idx]);
			float maxVal = std::max(intervalBuffers.second[idx], intervalBuffers.first[idx]);

			bool isNull = minVal==HORIZON_NULL_VALUE || maxVal==HORIZON_NULL_VALUE;
			if (isNull) {
				intervalBuffers.first[idx] = HORIZON_NULL_VALUE;
				intervalBuffers.second[idx] = HORIZON_NULL_VALUE;
			} else {
				intervalBuffers.first[idx] = minVal;
				intervalBuffers.second[idx] = maxVal;
			}
		}
	}

	if (intervalValid) {
		m_horizonDeltas.push_back(std::pair<float, float>(topDelta, bottomDelta));
		m_horizonNames.push_back(std::pair<QString, QString>(topPath, bottomPath));
		m_horizonIntervals.push_back(intervalBuffers);
	}

	return intervalValid;
}

void BnniJsonGenerator::defineLogsNames(const std::vector<BnniTrainingSet::BnniWellHeader>& headers) {
	if (m_wellBores.size()>0 && m_wellBores[0]->logsNames().size()!=headers.size()) {
		clearWellBores();
	}
	m_wellHeaders = headers;
}

template<typename SeismicType>
struct ReadSeismicAndResampleKernel {
	static void run(FILE* dataset, long header, long numTraces, long numSamples, int idxTraces, int idxProfils, double newSampleRate, double oldSampleRate, std::vector<double>& inBuffer, std::vector<double>& outBuffer) {
		//dataset->readTraceBlockAndSwap(inBuffer.data(), idxTraces, idxTraces+1, idxProfils);
		fseek(dataset, header + numSamples*(idxProfils*numTraces + idxTraces)*sizeof(SeismicType), SEEK_SET);
		fread(inBuffer.data(), numSamples, sizeof(SeismicType), dataset); // because sizeof(double)>=sizeof(SeismicType)

		SeismicType* buffer = static_cast<SeismicType*>(static_cast<void*>(inBuffer.data()));

		for (long idx=numSamples-1; idx>=0; idx--) {
			// swap
			SeismicType val = buffer[idx];
			char tmp;
			char* it1 = (char*) &val;
			char* it2 = (char*) ((&val)+1);
			it2--;
			while (it1<it2) {
				tmp = *it1;
				*it1 = *it2;
				*it2 = tmp;
				it1++;
				it2--;
			}

			inBuffer[idx] = val;
		}

		resampleSpline(newSampleRate, oldSampleRate, inBuffer, outBuffer);
	}
};

std::pair<bool, QString> BnniJsonGenerator::run() {
	bool ok = m_paths.size()>0 && m_wellBores.size()>0 && m_wellBores[0]->logsNames().size()>0;
	QString errorMsg;

	if (!ok) {
		errorMsg = "Initialization not valid";
		qDebug() << "BNNI : " + errorMsg;
		return std::pair<bool, QString>(ok, errorMsg);
	}

	std::vector<std::list<WellSeismicVoxel>> voxels;

	voxels.resize(m_wellBores.size());

	QStringList invalidLogName;
	QStringList invalidLogWell;
	QStringList invalidTfpName;
	QStringList invalidTfpWell;
	QStringList emptyExtractionWell;
	for (int iWell=0; iWell<m_wellBores.size(); iWell++) {
		WellBore* wellBore = m_wellBores[iWell];
		int nLog = wellBore->logsNames().size();
		bool wellValid = true;
		int badLogIdx = -1;
		for (int iLog=0; iLog<nLog; iLog++) {
			bool logChangeDone = wellBore->selectLog(iLog);
			if (!logChangeDone) {
				wellValid = false;
				badLogIdx = iLog;
				qDebug() << "BNNI : Failed to load log : " << wellBore->logsNames()[iLog] << "from well : " << wellBore->name();
				break;
			}
			if (m_useBandPassHighFrequency) {
				wellBore->activateFiltering(m_bandPassHighFrequency);
			} else {
				wellBore->deactivateFiltering();
			}

			const Logs& logs = wellBore->currentLog();
			extractSingleWellLog(logs, iLog, voxels[iWell], wellBore);
		}

		if (!wellValid) {
			ok = false;
			//errorMsg = "Failed to load well : " + wellBore->name();
			qDebug() << "BNNI : " + errorMsg;
			voxels[iWell].clear();

			invalidLogWell << wellBore->wellHead()->name() + " " + wellBore->name();
			if (badLogIdx>=0 && badLogIdx<wellBore->logsNames().size()) {
				invalidLogName << wellBore->logsNames()[badLogIdx];
			} else {
				invalidLogName << "No Log";
			}
			//break; // choose to stop process, it could be decided to go on without the faulty well
		} else if (m_seismicUnit==SampleUnit::TIME && !wellBore->isTfpDefined()) {
			invalidTfpWell << wellBore->wellHead()->name() + " " + wellBore->name();
			QString tfpName;
			if (wellBore->isTfpDefined()) {
				tfpName = wellBore->getTfpName();
			} else {
				tfpName = "No TFP";
			}
			invalidTfpName << tfpName;

			// if m_seismicUnit==SampleUnit::TIME && !wellBore->isTfpDefined() then voxels[iWell].size()==0, add security clear
			voxels[iWell].clear();
		} else if (voxels[iWell].size()==0) {
			emptyExtractionWell << wellBore->wellHead()->name() + " " + wellBore->name();
		}

		// clean
		long N = voxels[iWell].size();
		std::list<WellSeismicVoxel>::const_iterator it = voxels[iWell].begin();
		for (long i=0; i<N; i++) {
			bool valid = true;
			int j = 0;
			while (j<it->logs.size() && valid) {
				valid = it->logs[j].isDefined;
				j++;
			}
			std::list<WellSeismicVoxel>::const_iterator nextIt = it;
			nextIt++;
			if (!valid) {
				voxels[iWell].erase(it);
			}
			it = nextIt;
		}
	}

	std::vector<std::shared_ptr<BnniWell>> allWells;
	for (int iWell=0; iWell<m_wellBores.size(); iWell++) {
		std::shared_ptr<BnniWell> well(new OriginWell(m_wellBores[iWell]));
		//std::shared_ptr<BnniWell> wellAbstract = std::dynamic_pointer_cast<BnniWell>(well);
		allWells.push_back(well);
	}
	if (m_useAugmentation) {
		augmentData(voxels, allWells);
	}

	long sampleCount = 0;
	for (const std::list<WellSeismicVoxel>& voxelList : voxels) {
		sampleCount += voxelList.size();
	}

	ok = ok && invalidLogWell.size()==0 && invalidTfpWell.size()==0;
	if (ok) {
		qDebug() << "Successfully extracted voxels position for well bores, samples count : " << sampleCount;
	} else {
		qDebug() << "Extracted voxels position for well bores, samples count : " << sampleCount;
	}

	errorMsg = "Extracted " + QString::number(sampleCount) + " samples";
	if (invalidLogWell.size()<=10) {
		for (int i=0; i<invalidLogWell.size(); i++) {
			errorMsg += "\nWell Log invalid,  well : " + invalidLogWell[i] + ", log : " + invalidLogName[i];
		}
	} else {
		errorMsg += "\nToo many invalid Well Log, check your data";
	}
	if (invalidTfpWell.size()<=10) {
		for (int i=0; i<invalidTfpWell.size(); i++) {
			errorMsg += "\nWell TFP invalid,  well : " + invalidTfpWell[i] + ", tfp : " + invalidTfpName[i];
		}
	} else {
		errorMsg += "\nToo many invalid Well TFP, check your data";
	}
	if (emptyExtractionWell.size()>0) {
		for (int i=0; i<emptyExtractionWell.size(); i++) {
			errorMsg += "\nNo samples extracted from well : " + emptyExtractionWell[i];
		}
	}

	// get all random points
	std::map<long, QPoint> traceList;
	for (const std::list<WellSeismicVoxel>& voxelList : voxels) {
		for (const WellSeismicVoxel& voxel : voxelList) {
			long index = voxel.j + voxel.k * m_numTraces;
			traceList[index] = QPoint(voxel.j, voxel.k);
		}
	}

	// extract random
	std::map<long, std::vector<std::vector<double>>> randomBuffer;

	std::vector<double> iBuffer;
	iBuffer.resize(m_numSamples);
	for (int idx=0; idx<m_paths.size(); idx++) {
		FILE* dataset = fopen(m_paths[idx].toStdString().c_str(), "r");
		inri::Xt xt(m_paths[idx].toStdString());

		long header = xt.header_size();

		ImageFormats::QSampleType sampleType = Seismic3DAbstractDataset::translateType(xt.type());
		for (const std::pair<long, QPoint>& point : traceList) {
			if (idx==0) {
				std::vector<std::vector<double>> allBuf;
				std::vector<double> singleBuf;
				singleBuf.resize(m_numSamplesSurrechantillon);
				allBuf.resize(m_paths.size(), singleBuf);
				randomBuffer[point.first] = allBuf;
			}


			SampleTypeBinder binder(sampleType);
			binder.bind<ReadSeismicAndResampleKernel>(dataset, header, m_numTraces, m_numSamples, point.second.x(), point.second.y(),
					m_pasSampleSurrechantillon, xt.stepSamples(), iBuffer, randomBuffer[point.first][idx]);
		}
		fclose(dataset);
	}

	std::vector<std::vector<std::vector<double>>> seismicForVoxels;
	seismicForVoxels.resize(voxels.size());

	for (long wellIdx=0; wellIdx<voxels.size(); wellIdx++) {
		seismicForVoxels[wellIdx].resize(voxels[wellIdx].size());
		for (long pointIdx=0; pointIdx<voxels[wellIdx].size(); pointIdx++) {
			seismicForVoxels[wellIdx][pointIdx].resize((2*m_halfWindow+1)*m_paths.size());
			std::list<WellSeismicVoxel>::iterator it = voxels[wellIdx].begin();
			std::advance(it, pointIdx);
			const WellSeismicVoxel& voxel = *it;//voxels[wellIdx][pointIdx];
			long id = voxel.j + voxel.k * m_numTraces;

			const std::vector<std::vector<double>>& traceBuffers = randomBuffer[id];

			for (long iSeismic=0; iSeismic<m_paths.size(); iSeismic++) {
				for (long i=0; i<m_halfWindow*2+1; i++) {
					seismicForVoxels[wellIdx][pointIdx][i + iSeismic*(m_halfWindow*2+1)] = traceBuffers[iSeismic][voxel.i-m_halfWindow+i];
				}
			}
		}
	}

	createJSON(seismicForVoxels, voxels, allWells);

	return std::pair<bool, QString>(ok, errorMsg);
}

std::pair<bool, BnniJsonGenerator::IJKPoint> BnniJsonGenerator::isPointInBoundingBox(WellUnit unit, double logKey, WellBore* wellBore,
		WellShiftedGenerator* opGenerator) {
	IJKPointDouble pt;
	bool out = false;

	// get sampleI
	double sampleI;
	sampleI = wellBore->getDepthFromWellUnit(logKey, unit, m_seismicUnit, &out);

	// check i
	if (out) {
		double i;
		m_sampleTransformSurrechantillon->indirect(sampleI, i);
		pt.i = i;

		if (opGenerator==nullptr) {
			out = static_cast<int>(pt.i)-m_halfWindow>=0 && static_cast<int>(pt.i)+m_halfWindow<m_numSamplesSurrechantillon;
		}
	}

	// get and check jk
	if (out) {
		double x = wellBore->getXFromWellUnit(logKey, unit, &out);
		double y;
		if (out) {
			y = wellBore->getYFromWellUnit(logKey, unit, &out);
		}
		if (out) {
			double iMap, jMap;
			m_ijToXYTransfo->worldToImage(x, y, iMap, jMap);
			pt.j = iMap;
			pt.k = jMap;
			if (opGenerator) {
				std::shared_ptr<WellModifierOperator> op = opGenerator->getOperator(iMap, jMap);
				if (op!=nullptr) {
					pt = op->convert(pt, out);
				} else {
					out = false;
				}
			}
			out = out && static_cast<int>(pt.j)>=0 && static_cast<int>(pt.j)<m_numTraces &&
					static_cast<int>(pt.k)>=0 && static_cast<int>(pt.k)<m_numProfils;
			if (opGenerator) {
				out = out && static_cast<int>(pt.i)-m_halfWindow>=0 && static_cast<int>(pt.i)+m_halfWindow<m_numSamplesSurrechantillon;
			}

			if (m_horizonIntervals.size()>0) {
				// search if point is in an interval
				out = false;
				std::list<std::pair<std::vector<float>, std::vector<float>>>::const_iterator intervalIt = m_horizonIntervals.begin();
				std::size_t mapIdx = static_cast<int>(pt.j) + static_cast<int>(pt.k)*m_numTraces;
				while (!out && intervalIt!=m_horizonIntervals.end()) {
					float topVal = intervalIt->first[mapIdx];
					float bottomVal = intervalIt->second[mapIdx];
					out = topVal!=HORIZON_NULL_VALUE && bottomVal!=HORIZON_NULL_VALUE &&
							sampleI >= topVal && sampleI <= bottomVal;
					intervalIt++;
				}
			}
		}
	}

	IJKPoint ptInt;
	ptInt.i = pt.i;
	ptInt.j = pt.j;
	ptInt.k = pt.k;

	return std::pair<bool, IJKPoint>(out, ptInt);
}

std::pair<std::list<BnniJsonGenerator::WellSeismicVoxel>::iterator, int> BnniJsonGenerator::findVoxel(std::list<WellSeismicVoxel>& extractionList, std::list<WellSeismicVoxel>::iterator begin, const IJKPoint& voxel) {
	int step = 0;
	while (begin!=extractionList.end() && (begin->i!=voxel.i || begin->j!=voxel.j || begin->k!=voxel.k)) {
		begin++;
		step++;
	}
	return std::pair<std::list<WellSeismicVoxel>::iterator, int>(begin, step);
}

void BnniJsonGenerator::extractSingleWellLog(const Logs& log, int globalLogIndex, std::list<WellSeismicVoxel>& extractionList, WellBore* wellBore,
		WellShiftedGenerator* opGenerator) {
	if (log.unit==WellUnit::UNDEFINED_UNIT) {
		qDebug() << "BNNI : WellBore does not have transform to get X and Y without well unit";
		return;
	}
	int nLog = wellBore->logsNames().size();

	bool isFirstLog = extractionList.size()==0; // responsible for init
	bool lastValidDefined = false;
	bool voxelItDefined = false;
	std::list<WellSeismicVoxel>::iterator voxelIt = extractionList.begin();
	std::list<WellSeismicVoxel>::iterator lastValid = extractionList.begin();


	long globalIndex = 0;
	std::vector<double> sameVoxelLogVal;
	std::vector<double> sameVoxelMd;
	for (int intervalIndex=0; intervalIndex<log.nonNullIntervals.size(); intervalIndex++) {
		for (int logIndex = log.nonNullIntervals[intervalIndex].first; logIndex<=log.nonNullIntervals[intervalIndex].second; logIndex++) {
			std::pair<bool, IJKPoint> point = isPointInBoundingBox(log.unit, log.keys[logIndex], wellBore, opGenerator);
			bool inBoundingBox = point.first;
			IJKPoint voxelPoint = point.second;
			bool isValid = true;
			bool itMoved = false;
			int itMoveStep = 0;
			if (inBoundingBox) {
				bool createNewVoxel = extractionList.size()==0;
				if (!createNewVoxel && isFirstLog) {
					const WellSeismicVoxel& lastVoxel = extractionList.back();
					createNewVoxel = voxelPoint.i != lastVoxel.i || voxelPoint.j != lastVoxel.j || voxelPoint.k != lastVoxel.k;
				} else if (!isFirstLog) {
					std::pair<std::list<WellSeismicVoxel>::iterator, int> it = findVoxel(extractionList, voxelIt, voxelPoint);
					isValid = it.first!=extractionList.end();
					if (isValid) {
						itMoved = it.first!=voxelIt;
						if (itMoved) {
							lastValid = voxelIt;
							voxelIt = it.first;
							itMoveStep = it.second;
							lastValidDefined = voxelItDefined;
						}
						voxelItDefined = true;
					}
				}
				if (createNewVoxel || (!isFirstLog && isValid && itMoved)) {
					if (sameVoxelMd.size()>0) {
						WellSeismicVoxel* lastVoxel;
						if (isFirstLog) {
							lastVoxel = &(extractionList.back());
						} else {
							// remove stepped over
							if (itMoveStep>1) {
								std::list<WellSeismicVoxel>::iterator firstInvalid;
								if (lastValidDefined) {
									firstInvalid = lastValid;
									firstInvalid++;
								} else {
									firstInvalid = lastValid;
								}
								//extractionList.erase(firstInvalid, voxelIt);
							}

							lastVoxel = &(*voxelIt);
						}
						double md = 0;
						double N = sameVoxelLogVal.size();
						DefinedDouble attribute;
						attribute.val = 0;
						attribute.isDefined = N>0;

						for (std::size_t idx=0; idx<N; idx++) {
							md += sameVoxelMd[idx];
							attribute.val += sameVoxelLogVal[idx];
						}
						if (N>0) {
							md /= N;
							attribute.val /= N;
						}

						lastVoxel->logs[globalLogIndex] = attribute;
						lastVoxel->mds[globalLogIndex] = md;
					} else if (!isFirstLog) {
						if (itMoveStep>1) {
							std::list<WellSeismicVoxel>::iterator firstInvalid;
							if (lastValidDefined) {
								firstInvalid = lastValid;
								firstInvalid++;
							} else {
								firstInvalid = lastValid;
							}
							//extractionList.erase(firstInvalid, voxelIt);
						}
					}

					if (createNewVoxel) {
						WellSeismicVoxel newVoxel;
						newVoxel.i = voxelPoint.i;
						newVoxel.j = voxelPoint.j;
						newVoxel.k = voxelPoint.k;
						newVoxel.mds.resize(nLog, 0);
						DefinedDouble defaultVal;
						defaultVal.isDefined = false;
						newVoxel.logs.resize(nLog, defaultVal);
						extractionList.push_back(newVoxel);
					}
					sameVoxelLogVal.clear();
					sameVoxelMd.clear();
				}
				if (isFirstLog || (!isFirstLog && isValid)) {
					bool ok;
					double mdVal = wellBore->getMdFromWellUnit(log.keys[logIndex], log.unit, &ok);
					double logVal;
					if (ok) {
						logVal = wellBore->getLogFromWellUnit(log.keys[logIndex], log.unit, &ok);
					}
					if (ok) {
						sameVoxelMd.push_back(mdVal);
						sameVoxelLogVal.push_back(logVal);
						//sameVoxelLogVal.push_back(log.attributes[logIndex]);
					}
				}
			}
		}
	}
	if (sameVoxelMd.size()>0) {
		WellSeismicVoxel* lastVoxel;
		if (isFirstLog) {
			lastVoxel = &(extractionList.back());
		} else {
			lastVoxel = &(*voxelIt);
		}
		double md = 0;
		double N = sameVoxelLogVal.size();
		DefinedDouble attribute;
		attribute.val = 0;
		attribute.isDefined = N>0;

		for (std::size_t idx=0; idx<N; idx++) {
			md += sameVoxelMd[idx];
			attribute.val += sameVoxelLogVal[idx];
		}
		if (N>0) {
			md /= N;
			attribute.val /= N;
		}

		lastVoxel->logs[globalLogIndex] = attribute;
		lastVoxel->mds[globalLogIndex] = md;
	}
	voxelIt++;
	if (!isFirstLog && voxelIt!=extractionList.end()) {
		//extractionList.erase(voxelIt, extractionList.end());
	}
}

void BnniJsonGenerator::augmentData(std::vector<std::list<WellSeismicVoxel>>& voxels,
		std::vector<std::shared_ptr<BnniJsonGenerator::BnniWell>>& wells) {
	if (m_paths.size()==0) {
		return;
	}

	// add fake wells at the end of list voxels
	std::size_t Nwells = voxels.size();

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<> normalDistribution(0.0, m_gaussianNoiseStd);

	// test if there are some datasets in short
	QString pathShort;
	inri::Xt xt(m_paths[0].toStdString());
	bool isShort = xt.is_valid() && (xt.type()==inri::Xt::Type::Signed_16);
	if (isShort) {
		pathShort = m_paths[0];
	}

	if ((pathShort.isNull() || pathShort.isEmpty()) && m_useCnxAugmentation) {
		qDebug() << "CNX Augmentation activated but not short datasets to use. Skip augmentation.";
		return;
	}

	bool useBasicAugmentation = !m_useCnxAugmentation;

	std::shared_ptr<WellModifierOperatorGenerator> operatorGenerator = WellModifierOperatorGenerator::getGenerator({pathShort},
			m_augmentationDistance, m_pasSampleSurrechantillon);

	std::size_t voxelsIndex = 0;
	while (voxelsIndex<Nwells) {
		for (int iz=-1; iz<2; iz++) {
			for(int iy=-1; iy<2; iy++) {
				if (iz==0 && iy==0) {
					// skip origin well
					continue;
				}

				if (useBasicAugmentation) {
					// this is a reference, after a push_back it become invalid
					// thus to avoid copies, the reference need to be rechecked at each iteration of the for loop
					const std::list<WellSeismicVoxel>& originWell = voxels[voxelsIndex];

					voxels.push_back(originWell);
					for (std::list<WellSeismicVoxel>::iterator voxelIt = voxels.back().begin();
							voxelIt!=voxels.back().end(); voxelIt++) {
						voxelIt->j += iy * m_augmentationDistance;
						voxelIt->k += iz * m_augmentationDistance;
						for (std::size_t logIdx=0; logIdx<voxelIt->logs.size(); logIdx++) {
							double noise = normalDistribution(generator);
							voxelIt->logs[logIdx].val += noise;
						}
					}
				} else {
					WellBore* wellBore = m_wellBores[voxelsIndex];
					WellShiftedGenerator shiftedOpGenerator(operatorGenerator, iy+1, iz+1);

					std::list<WellSeismicVoxel> newWell;
					voxels.push_back(newWell);
					int nLog = wellBore->logsNames().size();
					bool wellValid = true;
					for (int iLog=0; iLog<nLog; iLog++) {
						bool logChangeDone = wellBore->selectLog(iLog);
						if (!logChangeDone) {
							wellValid = false;
							qDebug() << "BNNI : Failed to load log : " << wellBore->logsNames()[iLog] << "from well : " << wellBore->name();
							break;
						}
						if (m_useBandPassHighFrequency) {
							wellBore->activateFiltering(m_bandPassHighFrequency);
						} else {
							wellBore->deactivateFiltering();
						}

						const Logs& logs = wellBore->currentLog();

						extractSingleWellLog(logs, iLog, voxels.back(), wellBore, &shiftedOpGenerator);
					}
					if (!wellValid) {
						voxels.back().clear();
					}
					//qDebug() << "Check well:" << wellBore->name() << ", ori size:" << voxels[voxelsIndex].size() << ", new size:" << voxels.back().size();

					for (std::list<WellSeismicVoxel>::iterator voxelIt = voxels.back().begin();
							voxelIt!=voxels.back().end(); voxelIt++) {
						for (std::size_t logIdx=0; logIdx<voxelIt->logs.size(); logIdx++) {
							double noise = normalDistribution(generator);
							voxelIt->logs[logIdx].val += noise;
						}
					}
				}

				std::shared_ptr<AugmentedWell> augmentedWell = std::make_shared<AugmentedWell>();
				augmentedWell->setDy(iy * m_augmentationDistance);
				augmentedWell->setDz(iz * m_augmentationDistance);
				augmentedWell->setOriginWellName(m_wellBores[voxelsIndex]->name());
				augmentedWell->setOriginWellDescPath(m_wellBores[voxelsIndex]->getDescPath());
				augmentedWell->setOriginWellLogNames(m_wellBores[voxelsIndex]->logsNames());
				augmentedWell->setOriginWellLogNames(m_wellBores[voxelsIndex]->extractLogsKinds());
				augmentedWell->setOriginWellLogPaths(m_wellBores[voxelsIndex]->logsFiles());
				wells.push_back(std::dynamic_pointer_cast<BnniWell>(augmentedWell));
			}
		}

		voxelsIndex++;
	}
}

int BnniJsonGenerator::halfWindow() const {
	return m_halfWindow;
}

void BnniJsonGenerator::setHalfWindow(int val) {
	m_halfWindow = val;
}

void BnniJsonGenerator::createJSON(const std::vector<std::vector<std::vector<double>>>& seismicBuffers,
		const std::vector<std::list<BnniJsonGenerator::WellSeismicVoxel>>& voxels,
		const std::vector<std::shared_ptr<BnniWell>>& allWells) {
	// create document
	WDocument document;
	document.SetObject();

	nlohmann::json newDoc;

	// create first member : extractionParameters
	defineExtractionParameters(document, newDoc);

	// create second member : seismicParameters
	defineSeismicParameters(document, newDoc);

	// create third member : logsParameters
	defineLogsParameters(document, newDoc, allWells);

	// create fourth member : samples
	defineSamples(document, newDoc, seismicBuffers, voxels, allWells);

	qDebug() << "Is document valid" << document.IsObject();

	// create dir
	fs::path jsonPath(m_outputJsonFile.toStdString());
	fs::path searchPath = jsonPath.parent_path();
	bool dirExists = fs::exists(searchPath);
	bool valid = true;
	QStringList dirsToCreate;
	while (!dirExists && valid) {
		dirsToCreate.insert(0, QString(searchPath.filename().c_str()));
		valid = searchPath.has_parent_path();
		if (valid) {
			searchPath = searchPath.parent_path();
			dirExists = fs::exists(searchPath);
		}
	}
	if (dirExists && valid && dirsToCreate.count()>0) {
		QDir searchDir(QString(searchPath.c_str()));
		valid = searchDir.mkpath(dirsToCreate.join(QDir::separator()));
	}

	// write document to file system
	std::ofstream ofs(m_outputJsonFile.toStdString());
	rapidjson::OStreamWrapper osw(ofs);

	rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
	document.Accept(writer);

	// write bson
	QFileInfo jsonFileInfo(m_outputJsonFile);
	QString bsonFilePath = jsonFileInfo.dir().absoluteFilePath(jsonFileInfo.completeBaseName() + ".ubjson");

	QSaveFile bsonFile(bsonFilePath);
	if (bsonFile.open(QIODevice::WriteOnly)) {
		std::vector<std::uint8_t> v_bson = nlohmann::json::to_ubjson(newDoc);
		long long toWrite = v_bson.size();
		long long offset = 0;
		bool ok = true;
		while (ok && toWrite>0) {
			long long written = bsonFile.write(static_cast<char*>(static_cast<void*>(v_bson.data()))+offset, toWrite);
			ok = written >= 0;
			if (ok) {
				offset += written;
				toWrite -= written;
			}
		}
		if (ok) {
			bsonFile.commit();
		} else {
			bsonFile.cancelWriting();
		}
	}
}

void BnniJsonGenerator::defineExtractionParameters(WDocument& document, nlohmann::json& newDoc) {
	WValue extractionValue;
	extractionValue.SetObject();

	nlohmann::json extractionValueUbjsonObj = nlohmann::json::object();
	if (m_horizonIntervals.size()==0) {
		extractionValue.AddMember("bottomHorizonShift", 0.0f, document.GetAllocator());
		extractionValue.AddMember("bottomHorizon", "null", document.GetAllocator());
		extractionValue.AddMember("topHorizonShift", 0.0f, document.GetAllocator());
		extractionValue.AddMember("topHorizon", "null", document.GetAllocator());

		extractionValueUbjsonObj = { {"bottomHorizonShift", 0.0f}, {"bottomHorizon", "null"}, {"topHorizonShift", 0.0f}, {"topHorizon", "null"}};
	} else {
		// retro compatibility, warning naming can be wrong, it may need to check Sismage way of writing
		// but the horizons in NextVision may not exist in Sismage
		std::string bottomStr =  m_horizonNames[0].second.toStdString();
		extractionValue.AddMember("bottomHorizonShift", m_horizonDeltas[0].second, document.GetAllocator());
		extractionValue.AddMember("bottomHorizon", WValue().SetString(bottomStr.c_str(), document.GetAllocator()), document.GetAllocator());
		std::string topStr =  m_horizonNames[0].first.toStdString();
		extractionValue.AddMember("topHorizonShift", m_horizonDeltas[0].first, document.GetAllocator());
		extractionValue.AddMember("topHorizon", WValue().SetString(topStr.c_str(), document.GetAllocator()), document.GetAllocator());

		extractionValueUbjsonObj = { {"bottomHorizonShift", m_horizonDeltas[0].second}, {"bottomHorizon", bottomStr},
				{"topHorizonShift", m_horizonDeltas[0].first}, {"topHorizon", topStr}};

		if (m_horizonIntervals.size()>1) {
			WValue extractionValueArray;
			extractionValueArray.SetArray();

			nlohmann::json extractionValueUbjsonArray = nlohmann::json::array();
			for (int i=1; i<m_horizonNames.size(); i++) {
				WValue extractionValueExtra;
				extractionValueExtra.SetObject();

				std::string bottomStr = m_horizonNames[i].second.toStdString();
				extractionValueExtra.AddMember("bottomHorizonShift", m_horizonDeltas[i].second, document.GetAllocator());
				extractionValueExtra.AddMember("bottomHorizon", WValue().SetString(bottomStr.c_str(), document.GetAllocator()), document.GetAllocator());
				std::string topStr = m_horizonNames[i].first.toStdString();
				extractionValueExtra.AddMember("topHorizonShift", m_horizonDeltas[i].first, document.GetAllocator());
				extractionValueExtra.AddMember("topHorizon", WValue().SetString(topStr.c_str(), document.GetAllocator()), document.GetAllocator());

				extractionValueArray.PushBack(extractionValueExtra, document.GetAllocator());

				extractionValueUbjsonArray.push_back({ {"bottomHorizonShift", m_horizonDeltas[i].second}, {"bottomHorizon", bottomStr},
					{"topHorizonShift", m_horizonDeltas[i].first}, {"topHorizon", topStr}});
			}

			extractionValue.AddMember("otherHorizonIntervals", extractionValueArray, document.GetAllocator());

			extractionValueUbjsonObj["otherHorizonIntervals"] = extractionValueUbjsonArray;
		}
	}

	extractionValue.AddMember("depthUnit", WValue().SetString(m_depthUnit->getName().toStdString().c_str(), document.GetAllocator()), document.GetAllocator());

	extractionValueUbjsonObj["depthUnit"] = m_depthUnit->getName().toStdString();

	document.AddMember(WValue("extractionParameters"), extractionValue, document.GetAllocator());

	newDoc["extractionParameters"] = extractionValueUbjsonObj;
}

void BnniJsonGenerator::defineSeismicParameters(WDocument& document, nlohmann::json& newDoc) {
	WValue seismicParameters;
	seismicParameters.SetObject();
	seismicParameters.AddMember("halfWindowHeight", m_halfWindow, document.GetAllocator());

	nlohmann::json seismicParametersObj;
	seismicParametersObj["halfWindowHeight"] = m_halfWindow;

	WValue datasetsArray;
	datasetsArray.SetArray();

	nlohmann::json datasetsNewArray = nlohmann::json::array();
	for (int i=0; i<m_paths.size(); i++) {
		WValue datasetValue;
		datasetValue.SetObject();

		nlohmann::json datasetObject;

		WValue datasetDynamic;
		datasetDynamic.SetArray();
		datasetDynamic.PushBack(m_seismicDynamics[i].first, document.GetAllocator());
		datasetDynamic.PushBack(m_seismicDynamics[i].second, document.GetAllocator());

		nlohmann::json datasetDynamicArray = m_seismicDynamics[i];

		datasetValue.AddMember("dynamic", datasetDynamic, document.GetAllocator());

		datasetValue.AddMember("weight", m_seismicWeights[i], document.GetAllocator());

		datasetValue.AddMember("samplingRate", m_pasSampleSurrechantillon, document.GetAllocator());

		std::string sampleUnit;
		if (m_seismicUnit==SampleUnit::DEPTH) {
			sampleUnit = "depth";
		} else if (m_seismicUnit==SampleUnit::TIME) {
			sampleUnit = "time";
		} else {
			sampleUnit = "no_unit";
		}
		datasetValue.AddMember("sampleUnit", WValue().SetString(sampleUnit.c_str(), document.GetAllocator()), document.GetAllocator());
		datasetObject["dynamic"] = datasetDynamicArray;
		datasetObject["weight"] = m_seismicWeights[i];
		datasetObject["samplingRate"] = m_pasSampleSurrechantillon;
		datasetObject["sampleUnit"] = sampleUnit;

		QString surveyName; // survey dir name not the sismage name
		QString datasetName; // file name without "seismic3d." not the sismage name

		bool ok = getSeismicAndSurveyNames(m_paths[i], datasetName, surveyName);

		if (datasetName.isNull() || datasetName.isEmpty()) {
			datasetName = "UnknownDataset.xt";
			qDebug()<< "BNNI : Could not parse dataset name from file " << m_paths[i];
		}
		if (surveyName.isNull() || surveyName.isEmpty()) {
			surveyName = "UnknownSurvey";
			qDebug()<< "BNNI : Could not parse survey name from file " << m_paths[i];
		}

		std::string datasetJsonName = "Sismage Main\tIDL:DmSeismic/ExistingDataSet3DFactory:1.0\t" +
				surveyName.toStdString() + "/" + datasetName.toStdString();
		datasetValue.AddMember("dataset", WValue().SetString(datasetJsonName.c_str(), document.GetAllocator()), document.GetAllocator());

		datasetsArray.PushBack(datasetValue, document.GetAllocator());

		datasetObject["dataset"] = datasetJsonName;
		datasetsNewArray.push_back(datasetObject);
	}
	seismicParameters.AddMember("datasets", datasetsArray, document.GetAllocator());

	document.AddMember(WValue("seismicParameters"), seismicParameters, document.GetAllocator());

	seismicParametersObj["datasets"] = datasetsNewArray;
	newDoc["seismicParameters"] = seismicParametersObj;
}

void BnniJsonGenerator::defineLogsParameters(WDocument& document, nlohmann::json& newDoc, const std::vector<std::shared_ptr<BnniWell>>& allWells) {
	WValue logsParameters;
	logsParameters.SetObject();

	nlohmann::json logsParametersObj;

	if (m_useBandPassHighFrequency) {
		logsParameters.AddMember("bandPassHighFrequency", (float) m_bandPassHighFrequency, document.GetAllocator());
		logsParametersObj["bandPassHighFrequency"] = (float) m_bandPassHighFrequency;
	}

	logsParameters.AddMember("useDerivative", m_useDerivative, document.GetAllocator());
	logsParametersObj["useDerivative"] = m_useDerivative;

	WValue dynamicsArray;
	dynamicsArray.SetArray();
	nlohmann::json dynamicsNewArray = nlohmann::json::array();
	for (int i=0; i<m_wellHeaders.size(); i++) {
		WValue dynamics;
		dynamics.SetArray();
		dynamics.PushBack(m_wellHeaders[i].min, document.GetAllocator());
		dynamics.PushBack(m_wellHeaders[i].max, document.GetAllocator());
		dynamicsArray.PushBack(dynamics, document.GetAllocator());

		nlohmann::json dynamicsObj = nlohmann::json::array();
		dynamicsObj.push_back(m_wellHeaders[i].min);
		dynamicsObj.push_back(m_wellHeaders[i].max);

		dynamicsNewArray.push_back(dynamicsObj);
	}
	logsParameters.AddMember(WValue("logsDynamics"), dynamicsArray, document.GetAllocator());
	logsParametersObj["logsDynamics"] = dynamicsNewArray;

	WValue handpickedLogs;
	handpickedLogs.SetArray();
	nlohmann::json handpickedLogsArray = nlohmann::json::array();
	bool oneWellIsHandPicked = false;
	for (int i=0; i<allWells.size(); i++) {
		std::string firstStr = allWells[i]->getUniqueName().toStdString();
		for (int j=0; j<allWells[i]->getLogsCount(); j++) {
			if ((m_wellHeaders[j].filterType==BnniTrainingSet::WellName && allWells[i]->getLogName(j).compare(m_wellHeaders[j].filterStr)!=0) ||
					(m_wellHeaders[j].filterType==BnniTrainingSet::WellKind && allWells[i]->getLogKind(j).compare(m_wellHeaders[j].filterStr)!=0)) {
				WValue logArray;
				logArray.SetArray();

				// not sure if it is sismage name or folder name, may create issues later
				std::string thirdStr = allWells[i]->getLogUniqueName(j).toStdString();

				logArray.PushBack(WValue().SetString(firstStr.c_str(), document.GetAllocator()), document.GetAllocator());
				logArray.PushBack(j, document.GetAllocator());
				logArray.PushBack(WValue().SetString(thirdStr.c_str(), document.GetAllocator()), document.GetAllocator());

				handpickedLogs.PushBack(logArray, document.GetAllocator());

				oneWellIsHandPicked = true;

				nlohmann::json logNewArray = nlohmann::json::array();
				logNewArray.push_back(firstStr);
				logNewArray.push_back(j);
				logNewArray.push_back(thirdStr);
				handpickedLogsArray.push_back(logNewArray);
			}
		}
	}
	if (oneWellIsHandPicked) {
		logsParameters.AddMember(WValue("handpickedLogs"), handpickedLogs, document.GetAllocator());
		logsParametersObj["handpickedLogs"] = handpickedLogsArray;
	}

	WValue logsWeights;
	logsWeights.SetArray();
	nlohmann::json logsWeightsArray = nlohmann::json::array();
	for (int i=0; i<m_wellHeaders.size(); i++) {
		logsWeights.PushBack(m_wellHeaders[i].weight, document.GetAllocator());
		logsWeightsArray.push_back(m_wellHeaders[i].weight);
	}
	logsParameters.AddMember(WValue("logsWeights"), logsWeights, document.GetAllocator());
	logsParametersObj["logsWeights"] = logsWeightsArray;

	logsParameters.AddMember("logsCount", m_wellHeaders.size(), document.GetAllocator());
	logsParametersObj["logsCount"] = m_wellHeaders.size();

	WValue wellbores;
	wellbores.SetArray();
	nlohmann::json wellboresArray = nlohmann::json::array();
	for (int i=0; i<allWells.size(); i++) {
		std::string wellId = allWells[i]->getUniqueName().toStdString();
		wellbores.PushBack(WValue().SetString(wellId.c_str(), document.GetAllocator()), document.GetAllocator());
		wellboresArray.push_back(wellId);
	}
	logsParameters.AddMember("wellbores", wellbores, document.GetAllocator());
	logsParametersObj["wellbores"] = wellboresArray;

	WValue logColumns;
	logColumns.SetArray();
	nlohmann::json logColumnsArray = nlohmann::json::array();
	for (int i=0; i<m_wellHeaders.size(); i++) {
		WValue logValue;
		logValue.SetObject();
		WValue key;

		nlohmann::json logValueObj;
		std::string keyStr;
		if (m_wellHeaders[i].filterType==BnniTrainingSet::WellName) {
			key.SetString("name", document.GetAllocator());
			keyStr = "name";
		} else {
			key.SetString("kind", document.GetAllocator());
			keyStr = "kind";
		}
		logValue.AddMember(key, WValue().SetString(m_wellHeaders[i].filterStr.toStdString().c_str(), document.GetAllocator()), document.GetAllocator());
		logColumns.PushBack(logValue, document.GetAllocator());

		logValueObj[keyStr] = m_wellHeaders[i].filterStr.toStdString();
		logColumnsArray.push_back(logValueObj);
	}
	logsParameters.AddMember("logColumns", logColumns, document.GetAllocator());

	logsParameters.AddMember("mdSamplingRate", m_mdSamplingRate, document.GetAllocator());

	WValue wellboresShifts;
	wellboresShifts.SetObject(); // empty for now
	logsParameters.AddMember("wellboresShifts", wellboresShifts, document.GetAllocator());

	document.AddMember(WValue("logsParameters"), logsParameters, document.GetAllocator());

	logsParametersObj["logColumns"] = logColumnsArray;
	logsParametersObj["mdSamplingRate"] = m_mdSamplingRate;
	logsParametersObj["wellboresShifts"] = nlohmann::json::object({});
	newDoc["logsParameters"] = logsParametersObj;
}

void BnniJsonGenerator::defineSamples(WDocument& document, nlohmann::json& newDoc, 
		const std::vector<std::vector<std::vector<double>>>& seismicBuffers,
		const std::vector<std::list<BnniJsonGenerator::WellSeismicVoxel>>& voxels,
		const std::vector<std::shared_ptr<BnniWell>>& allWells) {
	WValue samples;
	samples.SetObject();

	nlohmann::json samplesObj = nlohmann::json::object({});
	for (long wellIndex=0; wellIndex<allWells.size(); wellIndex++) {
		BnniWell* wellBore = allWells[wellIndex].get();
		std::string wellBoreKeyStr = wellBore->getWellBoreSampleKey().toStdString();

		WValue key;
		key.SetString(wellBoreKeyStr.c_str(), document.GetAllocator());

		WValue array;
		array.SetArray();

		nlohmann::json newArray = nlohmann::json::array();
		std::list<WellSeismicVoxel>::const_iterator voxelIt = voxels[wellIndex].begin();
		long sampleIndex=0;
		while (sampleIndex<seismicBuffers[wellIndex].size()) {
			const WellSeismicVoxel& voxel = *voxelIt;

			WValue sample;
			sample.SetArray();
			nlohmann::json sampleArray = nlohmann::json::array();

			// add seismic values
			WValue seismicArray;
			seismicArray.SetArray();
			nlohmann::json seismicNewArray = nlohmann::json::array();

			const std::vector<std::vector<double>>& wellSeismics = seismicBuffers[wellIndex];
			const std::vector<double>& sampleSeismic = wellSeismics[sampleIndex];
			for (long seismicSampleIndex=0; seismicSampleIndex<sampleSeismic.size(); seismicSampleIndex++) {
				seismicArray.PushBack(sampleSeismic[seismicSampleIndex], document.GetAllocator());
				seismicNewArray.push_back(sampleSeismic[seismicSampleIndex]);
			}
			sample.PushBack(seismicArray, document.GetAllocator());
			sampleArray.push_back(seismicNewArray);

			// add logs values
			WValue logs;
			logs.SetArray();
			nlohmann::json logsArray = nlohmann::json::array();
			for (long logIndex=0; logIndex<voxel.logs.size(); logIndex++) {
				logs.PushBack(voxel.logs[logIndex].val, document.GetAllocator());
				logsArray.push_back(voxel.logs[logIndex].val);
			}
			sample.PushBack(logs, document.GetAllocator());
			sampleArray.push_back(logsArray);

			// add md
			double md = 0;
			for (double voxelMd : voxel.mds) {
				md += voxelMd;
			}
			if (voxel.mds.size()>0) {
				md /= voxel.mds.size();
			}
			sample.PushBack(md, document.GetAllocator());
			sampleArray.push_back(md);

			// add voxel.k voxel.j voxel.i
			WValue positions;
			positions.SetArray();
			positions.PushBack((float) voxel.k, document.GetAllocator());
			positions.PushBack((float) voxel.j, document.GetAllocator());
			positions.PushBack((float) voxel.i, document.GetAllocator());
			sample.PushBack(positions, document.GetAllocator());

			sampleArray.push_back({(float) voxel.k, (float) voxel.j, (float) voxel.i});

			array.PushBack(sample, document.GetAllocator());
			newArray.push_back(sampleArray);

			voxelIt++;
			sampleIndex++;
		}

		samples.AddMember(key, array, document.GetAllocator());
		samplesObj[wellBoreKeyStr] = newArray;
	}

	document.AddMember(WValue("samples"), samples, document.GetAllocator());
	newDoc["samples"] = samplesObj;
}

bool BnniJsonGenerator::getSeismicAndSurveyNames(const QString& datasetPath,
		QString& seismicName, QString& surveyName) {
	bool out = true;

	QFileInfo info(datasetPath);
	seismicName = info.completeSuffix(); // remove seismic3d.

	QDir dir = info.dir();
	out = dir.dirName().compare("SEISMIC")==0 && dir.cdUp() && dir.dirName().compare("DATA")==0 && dir.cdUp();

	if (out) {
		surveyName = dir.dirName();
	}

	return out;
}

QString BnniJsonGenerator::getWellBoreUniqueName(const WellBore* wellBore) {
	// not sure if it is sismage name or folder name, may create issues later
	QString outStr = "Sismage2 Main\tIDL:DmWell/ExistingWellBoreFactory:1.0\t";
	outStr += wellBore->getDirName();
	outStr += "||" + wellBore->wellHead()->getDirName();
	return outStr;
}

QString BnniJsonGenerator::getWellBoreUniqueName(const QString& wellHeadDirName, const QString& wellBoreDirName) {
	// not sure if it is sismage name or folder name, may create issues later
	QString outStr = "Sismage2 Main\tIDL:DmWell/ExistingWellBoreFactory:1.0\t";
	outStr += wellBoreDirName;
	outStr += "||" + wellHeadDirName;
	return outStr;
}

QString BnniJsonGenerator::getWellBoreSampleKey(const WellBore* wellBore) {
	// not sure if it is sismage name or folder name, may create issues later
	QString outStr = "Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0";
	outStr += wellBore->getDirName();
	outStr += "||" + wellBore->wellHead()->getDirName();
	return outStr;
}

QString BnniJsonGenerator::getWellBoreSampleKey(const QString& wellHeadDirName, const QString& wellBoreDirName) {
	// not sure if it is sismage name or folder name, may create issues later
	QString outStr = "Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0";
	outStr += wellBoreDirName;
	outStr += "||" + wellHeadDirName;
	return outStr;
}

QString BnniJsonGenerator::getWellBoreLogUniqueName(const WellBore* wellBore, int logIndex) {
	QString out = "Sismage2 Main\tIDL:DmWell/ExistingWellLogTraceFactory:1.0\t";
	out += wellBore->getLogFileName(logIndex) + "||" + wellBore->wellHead()->getDirName();
	out += "||" + wellBore->getDirName();
	return out;
}

QString BnniJsonGenerator::getWellBoreLogUniqueName(const QString& wellHeadDirName, const QString& wellBoreDirName,
		const QString& logFileName ) {
	QString out = "Sismage2 Main\tIDL:DmWell/ExistingWellLogTraceFactory:1.0\t";
	out += logFileName + "||" + wellHeadDirName;
	out += "||" + wellBoreDirName;
	return out;
}

QString BnniJsonGenerator::outputJsonFile() {
	return m_outputJsonFile;
}

void BnniJsonGenerator::setOutputJsonFile(const QString& newPath) {
	m_outputJsonFile = newPath;
}

double BnniJsonGenerator::pasSampleSurrechantillon() const {
	return m_pasSampleSurrechantillon;
}

void BnniJsonGenerator::setPasSampleSurrechantillon(double sampleRate) {
	m_pasSampleSurrechantillon = sampleRate;
}

float BnniJsonGenerator::mdSamplingRate() const {
	return m_mdSamplingRate;
}

void BnniJsonGenerator::setMdSamplingRate(float val) {
	m_mdSamplingRate = val;
}

bool BnniJsonGenerator::isActivatedBandPass() const {
	return m_useBandPassHighFrequency;
}

double BnniJsonGenerator::bandPassFrequency() const {
	return m_bandPassHighFrequency;
}

void BnniJsonGenerator::deactivateBandPass() {
	m_useBandPassHighFrequency = false;
}

void BnniJsonGenerator::activateBandPass(double freq) {
	m_useBandPassHighFrequency = true;
	m_bandPassHighFrequency = freq;
}

bool BnniJsonGenerator::useAugmentation() const {
	return m_useAugmentation;
}

void BnniJsonGenerator::setUseAugmentation(bool val) {
	m_useAugmentation = val;
}

int BnniJsonGenerator::augmentationDistance() const {
	return m_augmentationDistance;
}

void BnniJsonGenerator::setAugmentationDistance(int dist) {
	m_augmentationDistance = dist;
}

float BnniJsonGenerator::gaussianNoiseStd() const {
	return m_gaussianNoiseStd;
}

void BnniJsonGenerator::setGaussianNoiseStd(float val) {
	m_gaussianNoiseStd = val;
}

bool BnniJsonGenerator::useCnxAugmentation() const {
	return m_useCnxAugmentation;
}

void BnniJsonGenerator::toggleCnxAugmentation(bool val) {
	m_useCnxAugmentation = val;
}

bool BnniJsonGenerator::readHorizon(const QString& path, std::vector<float>& buffer) {
	bool horizonCompatible = m_paths.size()>0;
	if (!horizonCompatible) {
		return horizonCompatible;
	}

	QString surveyPath = QString::fromStdString(NextVisionDBManager::getSurvey3DPathFromHorizonPath(path.toStdString()));
	QString sismageName = QString::fromStdString(NextVisionDBManager::getSeismicSismageNameFromHorizonPath(path.toStdString()));

	QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(surveyPath, sismageName);

	horizonCompatible = !datasetPath.isNull() && !datasetPath.isEmpty();
	if (horizonCompatible) {
		inri::Xt xt(datasetPath.toStdString().c_str());
		horizonCompatible = xt.is_valid();
		if (horizonCompatible) {
			int nbTraces = xt.nRecords();
			int nbProfiles = xt.nSlices();
			float oriTraces = xt.startRecord();
			float pasTraces = xt.stepRecords();
			float oriProfiles = xt.startSlice();
			float pasProfiles = xt.stepSlices();

			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(datasetPath);
			SampleUnit unit = (timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH;

			horizonCompatible = nbTraces==m_numTraces && nbProfiles==m_numProfils && oriProfiles==m_startProfil &&
					pasProfiles==m_stepProfils && oriTraces==m_startTrace && pasTraces==m_stepTraces &&
					unit==m_seismicUnit;
		}
	}

	if (horizonCompatible) {
		buffer.resize(m_numTraces*m_numProfils);

		FILE* file = fopen(path.toStdString().c_str(), "r");
		horizonCompatible = file!=nullptr;
		if (horizonCompatible) {
			fread(buffer.data(), sizeof(float), buffer.size(), file);
			fclose(file);
		}
	}

	return horizonCompatible;
}

void BnniJsonGenerator::applyDelta(std::vector<float>& buffer, float delta) {
	for (std::size_t i=0; i<buffer.size(); i++) {
		buffer[i] += delta;
	}
}

SampleUnit BnniJsonGenerator::seismicUnit() const {
	return m_seismicUnit;
}

const MtLengthUnit* BnniJsonGenerator::depthUnit() const {
	return m_depthUnit;
}

void BnniJsonGenerator::setDepthUnit(const MtLengthUnit* depthUnit) {
	m_depthUnit = depthUnit;
}

BnniJsonGenerator::BnniWell::~BnniWell() {

}

BnniJsonGenerator::OriginWell::OriginWell(const WellBore* wellBore) {
	m_wellBore = wellBore;
}

BnniJsonGenerator::OriginWell::~OriginWell() {

}

QString BnniJsonGenerator::OriginWell::getUniqueName() const {
	return BnniJsonGenerator::getWellBoreUniqueName(m_wellBore);
}

int BnniJsonGenerator::OriginWell::getLogsCount() const {
	return m_wellBore->logsNames().size();
}

QString BnniJsonGenerator::OriginWell::getLogName(int logIdx) const {
	QString out;
	const std::vector<QString>& logNames = m_wellBore->logsNames();
	if (logIdx<logNames.size() && logIdx>=0) {
		out = logNames[logIdx];
	}
	return out;
}

QString BnniJsonGenerator::OriginWell::getLogKind(int logIdx) const {
	QString out;
	const std::vector<QString>& logKinds = m_wellBore->extractLogsKinds();
	if (logIdx<logKinds.size() && logIdx>=0) {
		out = logKinds[logIdx];
	}
	return out;
}

QString BnniJsonGenerator::OriginWell::getLogUniqueName(int logIdx) const {
	return getWellBoreLogUniqueName(m_wellBore, logIdx);
}

QString BnniJsonGenerator::OriginWell::getWellBoreSampleKey() const {
	return BnniJsonGenerator::getWellBoreSampleKey(m_wellBore);
}

BnniJsonGenerator::AugmentedWell::AugmentedWell() {

}

BnniJsonGenerator::AugmentedWell::~AugmentedWell() {

}

QString BnniJsonGenerator::AugmentedWell::getUniqueName() const {
	QDir wellBoreDir = QFileInfo(m_originWellDescPath).dir();
	QDir wellHeadDir(wellBoreDir);
	wellHeadDir.cdUp();
	QString wellBoreDirName = wellBoreDir.dirName() + getAugmentationCode();
	QString wellHeadDirName = wellHeadDir.dirName() + getAugmentationCode();
	return BnniJsonGenerator::getWellBoreUniqueName(wellHeadDirName, wellBoreDirName);
}

int BnniJsonGenerator::AugmentedWell::getLogsCount() const {
	return m_originWellLogNames.size();
}

QString BnniJsonGenerator::AugmentedWell::getLogName(int logIdx) const {
	QString out;
	if (logIdx>=0 && logIdx<m_originWellLogNames.size()) {
		out = m_originWellLogNames[logIdx];
	}
	return out;
}

QString BnniJsonGenerator::AugmentedWell::getLogKind(int logIdx) const {
	QString out;
	if (logIdx>=0 && logIdx<m_originWellLogKinds.size()) {
		out = m_originWellLogKinds[logIdx];
	}
	return out;
}

QString BnniJsonGenerator::AugmentedWell::getLogUniqueName(int logIdx) const {
	if (logIdx<0 || logIdx>=m_originWellLogPaths.size()) {
		return QString();
	}

	QDir wellBoreDir = QFileInfo(m_originWellDescPath).dir();
	QDir wellHeadDir(wellBoreDir);
	wellHeadDir.cdUp();
	QString wellBoreDirName = wellBoreDir.dirName() + getAugmentationCode();
	QString wellHeadDirName = wellHeadDir.dirName() + getAugmentationCode();
	QString logFileName = QFileInfo(m_originWellLogPaths[logIdx]).baseName();

	return BnniJsonGenerator::getWellBoreLogUniqueName(wellHeadDirName, wellBoreDirName, logFileName);
}

QString BnniJsonGenerator::AugmentedWell::getWellBoreSampleKey() const {
	QDir wellBoreDir = QFileInfo(m_originWellDescPath).dir();
	QDir wellHeadDir(wellBoreDir);
	wellHeadDir.cdUp();
	QString wellBoreDirName = wellBoreDir.dirName() + getAugmentationCode();
	QString wellHeadDirName = wellHeadDir.dirName() + getAugmentationCode();
	return BnniJsonGenerator::getWellBoreSampleKey(wellHeadDirName, wellBoreDirName);
}

QString BnniJsonGenerator::AugmentedWell::getAugmentationCode() const {
	return "_" + QString::number(m_dy) + "_" + QString::number(m_dz);
}

void BnniJsonGenerator::AugmentedWell::setOriginWellName(const QString& name) {
	m_originWellName = name;
}

void BnniJsonGenerator::AugmentedWell::setOriginWellDescPath(const QString& path) {
	m_originWellDescPath = path;
}

void BnniJsonGenerator::AugmentedWell::setOriginWellLogNames(const std::vector<QString>& logNames) {
	m_originWellLogNames = logNames;
}

void BnniJsonGenerator::AugmentedWell::setOriginWellLogKinds(const std::vector<QString>& logKinds) {
	m_originWellLogKinds = logKinds;
}

void BnniJsonGenerator::AugmentedWell::setOriginWellLogPaths(const std::vector<QString>& logPaths) {
	m_originWellLogPaths = logPaths;
}

void BnniJsonGenerator::AugmentedWell::setDy(int val) {
	m_dy = val;
}

void BnniJsonGenerator::AugmentedWell::setDz(int val) {
	m_dz = val;
}

BnniJsonGenerator::WellModifierOperator::~WellModifierOperator() {

}

BnniJsonGenerator::ShiftStretchSqueezeOperator::ShiftStretchSqueezeOperator(double dj, double dk,
		const std::vector<double>& inputData, const std::vector<double>& outData) : m_dj(dj), m_dk(dk) {
	m_valid = inputData.size()>0 && outData.size()==inputData.size();

	double maxi = std::numeric_limits<double>::lowest();
	double mini = std::numeric_limits<double>::max();
	double outValMini = 0;
	double outValMaxi = 0;
	if (m_valid) {
		long i=0;
		while (m_valid && i<inputData.size()) {
			if (i>0) {
				m_valid = inputData[i] > inputData[i-1];
			}
			if (m_valid) {
				if (maxi<inputData[i]) {
					maxi = inputData[i];
					outValMaxi = outData[i];
				}
				if (mini>inputData[i]) {
					mini = inputData[i];
					outValMini = outData[i];
				}
				i++;
			}
		}
	}
	if (m_valid) {
		m_inputDataMin = mini;
		m_inputDataMax = maxi;

		m_outputValueForMin = outValMini;
		m_outputValueForMax = outValMaxi;

		if (outData.size()==1) {
			m_a = 0;
			m_b = outData[0];
			m_useGsl = false;
		} else if (outData.size()==2) {
			WellBore::getAffineFromList(inputData.data(), outData.data(), m_a, m_b);
			m_useGsl = false;
		} else {
			m_useGsl = true;
			m_acc = gsl_interp_accel_alloc();
			m_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, inputData.size());
			std::vector<double> inputDataModif;
			inputDataModif.resize(inputData.size());
			memcpy(inputDataModif.data(), inputData.data(), inputData.size()*sizeof(double));
			std::vector<double> outDataModif;
			outDataModif.resize(outData.size());
			memcpy(outDataModif.data(), outData.data(), outData.size()*sizeof(double));
			gsl_spline_init(m_spline_steffen, inputDataModif.data(), outDataModif.data(), inputData.size());
		}
	}
}

BnniJsonGenerator::ShiftStretchSqueezeOperator::~ShiftStretchSqueezeOperator() {
	if (m_useGsl) {
		if (m_acc) {
			gsl_interp_accel_free(m_acc);
		}
		if (m_spline_steffen) {
			gsl_spline_free(m_spline_steffen);
		}
	}
}

bool BnniJsonGenerator::ShiftStretchSqueezeOperator::isValid() const {
	return m_valid;
}

bool BnniJsonGenerator::ShiftStretchSqueezeOperator::isInGslBounds(double val) const {
	return m_valid && m_useGsl && val>=m_inputDataMin && val<=m_inputDataMax;
}

BnniJsonGenerator::IJKPointDouble BnniJsonGenerator::ShiftStretchSqueezeOperator::convert(IJKPointDouble pt, bool& ok) const {
	ok = m_valid;
	if (!ok) {
		IJKPointDouble outPt = pt;
		return outPt;
	}

	IJKPointDouble outPt;
	if (m_useGsl) {
		if (isInGslBounds(pt.i)) {
			outPt.i = gsl_spline_eval(m_spline_steffen, pt.i, m_acc);
		} else if (pt.i>m_inputDataMax) {
			outPt.i = m_outputValueForMax - m_inputDataMax + pt.i;
		} else if (pt.i<m_inputDataMin) {
			outPt.i = m_outputValueForMin - m_inputDataMin + pt.i;
		}
	} else {
		outPt.i = pt.i * m_a + m_b;
	}
	outPt.j = pt.j + m_dj;
	outPt.k = pt.k + m_dk;

	return outPt;
}

BnniJsonGenerator::MeanWellModifierOperator::MeanWellModifierOperator(const std::vector<std::shared_ptr<WellModifierOperator>>& ops) {
	m_ops = ops;
}

BnniJsonGenerator::MeanWellModifierOperator::~MeanWellModifierOperator() {

}

bool BnniJsonGenerator::MeanWellModifierOperator::isValid() const {
	bool valid = false;
	int i=0;
	while(!valid && i<m_ops.size()) {
		valid = m_ops[i]->isValid();
		i++;
	}
	return valid;
}

BnniJsonGenerator::IJKPointDouble BnniJsonGenerator::MeanWellModifierOperator::convert(IJKPointDouble pt, bool& ok) const {
	std::vector<IJKPointDouble> points;
	for (int i=0; i<m_ops.size(); i++) {
		bool valid = m_ops[i]->isValid();
		IJKPointDouble newPt;
		if (valid) {
			newPt = m_ops[i]->convert(pt, valid);
		}
		if (valid) {
			points.push_back(newPt);
		}
	}

	IJKPointDouble outPt;
	if (m_ops.size()==0) {
		ok = true;
		outPt = pt;
	} else if (points.size()==0) {
		ok = false;
	} else {
		outPt.i = 0;
		outPt.j = 0;
		outPt.k = 0;

		for (int i=0; i<points.size(); i++) {
			outPt.i += points[i].i;
			outPt.j += points[i].j;
			outPt.k += points[i].k;
		}
		outPt.i /= points.size();
		outPt.j /= points.size();
		outPt.k /= points.size();

	}
	return outPt;
}

std::shared_ptr<BnniJsonGenerator::WellModifierOperatorGenerator> BnniJsonGenerator::WellModifierOperatorGenerator::getGenerator(
		const std::vector<QString>& seismics, int augmentationDistance, double pasSampleSurrechantillon) {
	bool valid = seismics.size()>0;

	bool dimSet = false;
	int dimI;
	int dimJ;
	int dimK;
	int i=0;
	while (i<seismics.size()) {
		inri::Xt xt(seismics[i].toStdString());
		valid = xt.is_valid();
		if (valid) {
			if (dimSet) {
				valid = dimI == xt.nSamples() && dimJ == xt.nRecords() && dimK == xt.nSlices();
			} else {
				dimI = xt.nSamples();
				dimJ = xt.nRecords();
				dimK = xt.nSlices();
				dimSet = true;
			}
		}
		i++;
	}

	valid = valid && dimSet;
	std::shared_ptr<WellModifierOperatorGenerator> obj;
	if (valid) {
		obj = std::shared_ptr<WellModifierOperatorGenerator>(new WellModifierOperatorGenerator(
				seismics, dimJ, augmentationDistance, pasSampleSurrechantillon));
	}
	return obj;
}

BnniJsonGenerator::WellModifierOperatorGenerator::~WellModifierOperatorGenerator() {

}

std::shared_ptr<BnniJsonGenerator::WellModifierOperator> BnniJsonGenerator::WellModifierOperatorGenerator::getOperator(
		int baseJ, int baseK, int offsetJ, int offsetK) {
	long cacheId = baseJ + baseK * m_dimJ;
	auto itCache = m_cachedOperators.find(cacheId);
	if (itCache!=m_cachedOperators.end()) {
		int gridIdx = offsetJ + offsetK * 3;
		return itCache->second[gridIdx];
	}

	const int nGridPoints = 9;
	std::array<std::vector<std::shared_ptr<WellModifierOperator>>, nGridPoints> allOperators;
	for (int seismicIdx=0; seismicIdx<m_seismics.size(); seismicIdx++) {
		std::vector<std::vector<double>> patches = CnxPatchIndex::getBorderIndex(m_seismics[seismicIdx].toStdString(),
				m_pasSampleSurrechantillon, baseJ, baseK, m_augmentationDistance*2+1);

		if (patches.size()==nGridPoints && patches[nGridPoints/2].size()>0) {
			for (int i=0; i<3; i++) {
				for (int k=0; k<3; k++) {
					int gridIdx = i+k*3;
					if (i!=1 || k!=1) {
						allOperators[gridIdx].push_back(std::shared_ptr<ShiftStretchSqueezeOperator>(
								new ShiftStretchSqueezeOperator((i-1)*m_augmentationDistance, (k-1)*m_augmentationDistance,
								patches[nGridPoints/2], patches[gridIdx])));
					}
				}
			}
		}
	}

	std::array<std::shared_ptr<WellModifierOperator>, nGridPoints> meanOperators;
	for (int i=0; i<nGridPoints; i++) {
		meanOperators[i] = std::dynamic_pointer_cast<WellModifierOperator>(std::shared_ptr<MeanWellModifierOperator>(new MeanWellModifierOperator(allOperators[i])));
	}
	m_cachedOperators[cacheId] = meanOperators;
	int gridIdx = offsetJ + offsetK * 3;
	return meanOperators[gridIdx];
}

BnniJsonGenerator::WellModifierOperatorGenerator::WellModifierOperatorGenerator(const std::vector<QString>& seismics, int dimJ, int augmentationDistance,
		double pasSampleSurrechantillon) {
	m_seismics = seismics;
	m_dimJ = dimJ;
	m_augmentationDistance = augmentationDistance;
	m_pasSampleSurrechantillon = pasSampleSurrechantillon;
}

BnniJsonGenerator::WellShiftedGenerator::WellShiftedGenerator(std::shared_ptr<WellModifierOperatorGenerator> opGenerator, int offsetJ, int offsetK) {
	m_opGenerator = opGenerator;
	m_offsetJ = offsetJ;
	m_offsetK = offsetK;
}

BnniJsonGenerator::WellShiftedGenerator::~WellShiftedGenerator() {

}

std::shared_ptr<BnniJsonGenerator::WellModifierOperator> BnniJsonGenerator::WellShiftedGenerator::getOperator(int baseJ, int baseK) {
	return m_opGenerator->getOperator(baseJ, baseK, m_offsetJ, m_offsetK);
}
