#include "fixedrgblayersfromdatasetandcubeimplmulti.h"

#include "sismagedbmanager.h"
#include "smdataset3D.h"
#include "seismicsurvey.h"
#include "datasetrelatedstorageimpl.h"
#include "stringselectordialog.h"
#include "GeotimeProjectManagerWidget.h"
#include "sampletypebinder.h"
#include "gdalloader.h"
#include "Xt.h"

#include "gdal.h"
#include <gdal_priv.h>
#include <memory>

#include <QDir>
#include <QMessageBox>

FixedRGBLayersFromDatasetAndCubeImplMulti::FixedRGBLayersFromDatasetAndCubeImplMulti(QString dirPath,
		QString name, bool rgb1Active, WorkingSetManager *workingSet,
		const Grid3DParameter& params, QObject *parent) : FixedRGBLayersFromDatasetAndCube(name, workingSet, params, parent) {
//	loadIsoAndRgb(rgb2Path, rgb1Path);
	loadObjectParamsFromDir(dirPath, rgb1Active);

	initLayersList();

	setCurrentImageIndex(0);
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::getObjFile(int index) const
{
//	QFileInfo fileinfo(m_rgb2Path);
//	QString temp = fileinfo.completeBaseName();
//	qDebug()<<"fileinfo "<<temp;
//
//	QDir dir = fileinfo.absoluteDir();
//	if(!dir.exists(temp))
//	{
//		dir.mkdir(temp);
//	}
//
//	dir.cd(temp);

	QDir dir(m_dirPath);

	QString res = QString::number(index*isoStep()+isoOrigin())+".obj";//3do

	return dir.absoluteFilePath(res);
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::getIsoFileFromIndex(int index) const {
	QDir dir(m_dirPath);

	QString res = "iso_" + QString::number(index*isoStep()+isoOrigin())+".raw";//3do

	return dir.absoluteFilePath(res);
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::getRgb2FileFromIndex(int index) const {
	QDir dir(m_dirPath);

	QString res = "rgb2_" + QString::number(index*isoStep()+isoOrigin())+".raw";//3do

	return dir.absoluteFilePath(res);
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::getRgb1FileFromIndex(int index) const {
	QDir dir(m_dirPath);

	QString res = "rgb1_" + QString::number(index*isoStep()+isoOrigin())+".png";//3do

	return dir.absoluteFilePath(res);
}

FixedRGBLayersFromDatasetAndCubeImplMulti::~FixedRGBLayersFromDatasetAndCubeImplMulti() {

}

void FixedRGBLayersFromDatasetAndCubeImplMulti::loadObjectParamsFromDir(
		const QString& dirPath, bool useRgb1) {
	m_dirPath = dirPath;

	QDir dir(m_dirPath);

	QFileInfoList files = dir.entryInfoList(QStringList() << "iso_*.raw", QDir::Files | QDir::Readable);

	// files should be defined as a grid
	bool useRgb1Valid = useRgb1;
	std::vector<int> values;
	for (const QFileInfo& e : files) {
		QString strVal = e.baseName().split("_").last();

		bool ok;
		int val = strVal.toInt(&ok);
		if (ok) {
			values.push_back(val);
			if (useRgb1Valid) {
				useRgb1Valid = dir.exists("rgb1_"+strVal+".png");
			}
		}
	}

	std::sort(values.begin(), values.end());
	if (values.size()>0) {
		int min = values[0];
		int max = values[values.size()-1];

		// TODO deduce step from list "values"
		m_numLayers = std::abs(min-max) / std::abs(m_isoStep);
		m_isoOrigin = (m_numLayers-1) * (-m_isoStep);
		m_useRgb1 = useRgb1Valid;
	} else {
		qDebug() << "FixedRGBLayersFromDatasetAndCube : Invalid dirPath";
		m_numLayers = 0;
		m_useRgb1 = false;
		m_isoOrigin = 0;
	}
}

void FixedRGBLayersFromDatasetAndCubeImplMulti::getImageForIndex(long newIndex,
		CUDAImagePaletteHolder* redCudaBuffer, CUDAImagePaletteHolder* greenCudaBuffer,
		CUDAImagePaletteHolder* blueCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return;

	QMutexLocker locker(&m_lock);

	// read rgb
	std::size_t w = width();
	std::size_t h = depth();
	std::size_t layerSize = w * h;

	if (mode()==CACHE && ((newIndex-cacheFirstIndex())%cacheStepIndex())==0 && ((newIndex-cacheFirstIndex())/cacheStepIndex())>0 &&
			((newIndex-cacheFirstIndex())/cacheStepIndex())<((cacheLastIndex()-cacheFirstIndex())/cacheStepIndex())) {
		long cacheRelativeIndex = (newIndex-cacheFirstIndex())/cacheStepIndex();

		std::list<SurfaceCache>::iterator it = m_cacheList.begin();
		std::advance(it, cacheRelativeIndex);
		std::vector<short> rgbPlanar;
		const short* oriRgbData = static_cast<const short*>(static_cast<const void*>(it->rgb.constData()));
		rgbPlanar.resize(it->rgb.size());
		for (std::size_t pixelIdx=0; pixelIdx<layerSize; pixelIdx++) {
			rgbPlanar[pixelIdx] = oriRgbData[pixelIdx*3];
			rgbPlanar[pixelIdx + layerSize] = oriRgbData[pixelIdx*3+1];
			rgbPlanar[pixelIdx + layerSize*2] = oriRgbData[pixelIdx*3+2];
		}
		redCudaBuffer->updateTexture(rgbPlanar.data(), false, it->redRange);
		greenCudaBuffer->updateTexture(rgbPlanar.data() + layerSize, false, it->greenRange);
		blueCudaBuffer->updateTexture(rgbPlanar.data() + 2* layerSize, false, it->blueRange);
		isoCudaBuffer->updateTexture(it->iso.constData(), false);
	} else {
		std::vector<short> buf;
		std::vector<short> outBuf;
		buf.resize(layerSize*3);
		outBuf.resize(layerSize);
		//size_t absolutePosition = layerSize * newIndex * sizeof(short) * 4;
		{
//			QMutexLocker b2(&m_lockRgb2);
//			fseek(m_fRgb2, absolutePosition, SEEK_SET);
//			fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);
			FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
			if (isoFile!=NULL) {
//				fseek(isoFile, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize, isoFile);
				fclose(isoFile);
				isoCudaBuffer->updateTexture(buf.data(), false);
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
			}
		}
//		for (std::size_t idx=0; idx<layerSize; idx++) {
//			short val = buf[idx*4];
//			// swap
//			//swap(val);
//			outBuf[idx] = val;
//		}
//		redCudaBuffer->updateTexture(outBuf.data(), false);
//
//		for (std::size_t idx=0; idx<layerSize; idx++) {
//			short val = buf[idx*4+1];
//			//swap(val);
//			outBuf[idx] = val;
//		}
//		greenCudaBuffer->updateTexture(outBuf.data(), false);
//
//		for (std::size_t idx=0; idx<layerSize; idx++) {
//			short val = buf[idx*4+2];
//			//swap(val);
//			outBuf[idx] = val;
//		}
//		blueCudaBuffer->updateTexture(outBuf.data(), false);
//
//		for (std::size_t idx=0; idx<layerSize; idx++) {
//			short val = buf[idx*4+3];
//			//swap(val);
//			outBuf[idx] = val;
//		}
//		isoCudaBuffer->updateTexture(outBuf.data(), false);
		bool rgbValid = false;
		if (m_useRgb1) {
			// TODO
			bool ok = readRgb1(getRgb1FileFromIndex(newIndex), buf.data(), w, h);
			if (ok) {
				rgbValid = true;
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB1";
			}
		} else {
			FILE* rgb2File = fopen(getRgb2FileFromIndex(newIndex).toStdString().c_str(), "r");
			if (rgb2File!=NULL) {
//				fseek(rgb2File, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize*3, rgb2File);
				fclose(rgb2File);
				rgbValid = true;
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
			}
		}
		if (rgbValid) {
			std::vector<short> rgbPlanar;
			rgbPlanar.resize(layerSize*3);
			for (std::size_t pixelIdx=0; pixelIdx<layerSize; pixelIdx++) {
				rgbPlanar[pixelIdx] = buf[pixelIdx*3];
				rgbPlanar[pixelIdx + layerSize] = buf[pixelIdx*3+1];
				rgbPlanar[pixelIdx + layerSize*2] = buf[pixelIdx*3+2];
			}
			redCudaBuffer->updateTexture(rgbPlanar.data(), false);
			greenCudaBuffer->updateTexture(rgbPlanar.data() + layerSize, false);
			blueCudaBuffer->updateTexture(rgbPlanar.data() + layerSize*2, false);
		}
	}
}

bool FixedRGBLayersFromDatasetAndCubeImplMulti::getImageForIndex(long newIndex,
		QByteArray& rgbBuffer, QByteArray& isoBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return false;

	QMutexLocker locker(&m_lock);

	// read rgb
	std::size_t w = width();
	std::size_t h = depth();

	std::vector<short> buf;
	std::size_t layerSize = w * h;
	rgbBuffer.resize(layerSize*3 * sizeof(short));
	isoBuffer.resize(layerSize* sizeof(short));

	bool isValid = checkValidity<short>(rgbBuffer, layerSize*3);
	isValid = isValid && checkValidity<short>(isoBuffer, layerSize);

	if (isValid && mode()==CACHE && ((newIndex-cacheFirstIndex())%cacheStepIndex())==0 && ((newIndex-cacheFirstIndex())/cacheStepIndex())>0 &&
			((newIndex-cacheFirstIndex())/cacheStepIndex())<((cacheLastIndex()-cacheFirstIndex())/cacheStepIndex())) {
		long cacheRelativeIndex = (newIndex-cacheFirstIndex())/cacheStepIndex();

		std::list<SurfaceCache>::iterator it = m_cacheList.begin();
		std::advance(it, cacheRelativeIndex);
		rgbBuffer = it->rgb;
		isoBuffer = it->iso;
	} else if (isValid) {
//		buf.resize(layerSize*3);
//		isValid = checkValidity(buf, layerSize*3);

		if (isValid) {
			{
				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if (isoFile!=NULL) {
//					fseek(isoFile, absolutePosition, SEEK_SET);
					fread(isoBuffer.data(), sizeof(short), layerSize, isoFile);
					fclose(isoFile);
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
				}
			}
			if (m_useRgb1) {
				// TODO
				bool ok = readRgb1(getRgb1FileFromIndex(newIndex), static_cast<short*>(static_cast<void*>(rgbBuffer.data())), w, h);
				if (!ok) {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB1";
				}
			} else {
				FILE* rgb2File = fopen(getRgb2FileFromIndex(newIndex).toStdString().c_str(), "r");
				if (rgb2File!=NULL) {
					fread(rgbBuffer.data(), sizeof(short), layerSize*3, rgb2File);
					fclose(rgb2File);
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
				}
			}
//			size_t absolutePosition = layerSize * newIndex * sizeof(short) * 4;
//			{
//				QMutexLocker b2(&m_lockRgb2);
//				fseek(m_fRgb2, absolutePosition, SEEK_SET);
//				fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);
//			}
//			if (!m_useRgb1) {
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4];
//					// swap
//					//swap(val);
//					redBuffer[idx] = val;
//				}
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4+1];
//					//swap(val);
//					greenBuffer[idx] = val;
//				}
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4+2];
//					//swap(val);
//					blueBuffer[idx] = val;
//				}
//			}
//
//			for (std::size_t idx=0; idx<layerSize; idx++) {
//				short val = buf[idx*4+3];
//				//swap(val);
//				isoBuffer[idx] = val;
//			}
//
//			if (m_useRgb1) {
//				size_t absolutePositionRgb1 = layerSize * newIndex * sizeof(char) * 3;
//				{
//					QMutexLocker b1(&m_lockRgb1);
//					fseek(m_fRgb1, absolutePositionRgb1, SEEK_SET);
//					fread(buf.data(), sizeof(unsigned char), layerSize*3, m_fRgb1); // reuse buffer
//				}
//				unsigned char* bufAsChar = static_cast<unsigned char*>(static_cast<void*>(buf.data()));
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3];
//					redBuffer[idx] = val;
//				}
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3+1];
//					greenBuffer[idx] = val;
//				}
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3+2];
//					blueBuffer[idx] = val;
//				}
//			}
		}
	}
	return isValid;
}

void FixedRGBLayersFromDatasetAndCubeImplMulti::setCurrentImageIndexInternal(long newIndex) {
	if (newIndex<0 || newIndex>=m_numLayers) {
		m_currentImageIndex = -1;
		return;
	}
	if (m_currentImageIndex==newIndex) {
		return;
	} else {
		m_currentImageIndex = newIndex;
	}
	if (m_currentImageIndex!=-1) {
		// read rgb
		QMutexLocker locker(&m_lock);
		std::size_t w = width();
		std::size_t h = depth();

		if (mode()==CACHE && ((newIndex-cacheFirstIndex())%cacheStepIndex())==0 && ((newIndex-cacheFirstIndex())/cacheStepIndex())>=0 &&
					((newIndex-cacheFirstIndex())/cacheStepIndex())<=((cacheLastIndex()-cacheFirstIndex())/cacheStepIndex())) {
			long cacheRelativeIndex = (newIndex-cacheFirstIndex())/cacheStepIndex();
			//std::chrono::steady_clock::time_point preinit = std::chrono::steady_clock::now();
			std::list<SurfaceCache>::iterator it = m_cacheList.begin();
			std::advance(it, cacheRelativeIndex);
//			m_currentRGB->get(0)->updateTexture(it->red.data(), false, it->redRange);
//			m_currentRGB->get(1)->updateTexture(it->green.data(), false, it->greenRange);
//			m_currentRGB->get(2)->updateTexture(it->blue.data(), false, it->blueRange);
			//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			m_currentRGB->updateTexture(it->rgb, false, it->redRange, it->greenRange, it->blueRange,
					it->redHistogram, it->greenHistogram, it->blueHistogram);

			m_currentIso->updateTexture(it->iso, false);
			//QCoreApplication::processEvents();
			/*std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			qDebug() << "cache update iso : " << std::chrono::duration<double, std::milli>(end-start).count() <<
					", find cache" << std::chrono::duration<double, std::milli>(start-preinit).count();*/
		} else {
			//std::chrono::steady_clock::time_point initRead, isoRead, isoTexture, rgb1Read, resize;
			//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

			QByteArray buf, grayBuf;
//			std::vector<short> outBuf;
			std::size_t layerSize = w * h;
			buf.resize(layerSize*3*sizeof(short));
			grayBuf.resize(layerSize*sizeof(short));
//			outBuf.resize(layerSize);
			//resize = std::chrono::steady_clock::now();
			{
				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if (isoFile!=NULL) {
				//	initRead = std::chrono::steady_clock::now();
					fread(grayBuf.data(), sizeof(short), layerSize, isoFile);
					fclose(isoFile);
				//	isoRead = std::chrono::steady_clock::now();
					m_currentIso->updateTexture(grayBuf, false);
					//isoTexture = std::chrono::steady_clock::now();
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
				}
			}
			//rgb1Read = isoTexture;
			if (m_useRgb1) {
				// TODO
				short* bufShort = static_cast<short*>(static_cast<void*>(buf.data()));
				bool ok = readRgb1(getRgb1FileFromIndex(newIndex), bufShort, w, h);
				//rgb1Read = std::chrono::steady_clock::now();
				if (ok) {
					QVector2D range(0, 255);
					m_currentRGB->updateTexture(buf, false, range, range, range);
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB1";
				}
			} else {
				FILE* rgb2File = fopen(getRgb2FileFromIndex(newIndex).toStdString().c_str(), "r");
				if (rgb2File!=NULL) {
	//				fseek(rgb2File, absolutePosition, SEEK_SET);
					fread(buf.data(), sizeof(short), layerSize*3, rgb2File);
					fclose(rgb2File);
					m_currentRGB->updateTexture(buf, false);
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
				}
			}
		//	auto end = std::chrono::steady_clock::now();

		/*	qDebug() << "All : " << std::chrono::duration<double, std::milli>(end-start).count() <<
					", Resize : " << std::chrono::duration<double, std::milli>(resize-start).count() <<
					", Fopen : " << std::chrono::duration<double, std::milli>(initRead-resize).count() <<
					", Iso Read : " << std::chrono::duration<double, std::milli>(isoRead-initRead).count() <<
					", Iso update : " << std::chrono::duration<double, std::milli>(isoTexture-isoRead).count() <<
					", Rgb1 read : " << std::chrono::duration<double, std::milli>(rgb1Read-isoTexture).count() <<
					", Rgb1 update : "<< std::chrono::duration<double, std::milli>(end - rgb1Read).count();*/

//			size_t absolutePosition = layerSize * m_currentImageIndex * sizeof(short) * 4;
//			{
//				QMutexLocker b2(&m_lockRgb2);
//				fseek(m_fRgb2, absolutePosition, SEEK_SET);
//				fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);
//			}
//
//			if (!m_useRgb1) {
//				//#pragma omp parallel for
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4];
//					// swap
//					//swap(val);
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(0)->updateTexture(outBuf.data(), false);
//				//#pragma omp parallel for
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4+1];
//					//swap(val);
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(1)->updateTexture(outBuf.data(), false);
//				//#pragma omp parallel for
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					short val = buf[idx*4+2];
//					//swap(val);
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(2)->updateTexture(outBuf.data(), false);
//			}
//			//#pragma omp parallel for
//			for (std::size_t idx=0; idx<layerSize; idx++) {
//				short val = buf[idx*4+3];
//				//swap(val);
//				outBuf[idx] = val;
//			}
//			m_currentIso->updateTexture(outBuf.data(), false);
//
//			if (m_useRgb1) {
//				size_t absolutePositionRgb1 = layerSize * newIndex * sizeof(char) * 3;
//				{
//					QMutexLocker b1(&m_lockRgb1);
//					fseek(m_fRgb1, absolutePositionRgb1, SEEK_SET);
//					fread(buf.data(), sizeof(unsigned char), layerSize*3, m_fRgb1); // reuse buffer
//				}
//				unsigned char* bufAsChar = static_cast<unsigned char*>(static_cast<void*>(buf.data()));
//
//				QVector2D range(0, 255);
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3];
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(0)->updateTexture(outBuf.data(), false, range);
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3+1];
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(1)->updateTexture(outBuf.data(), false, range);
//
//				for (std::size_t idx=0; idx<layerSize; idx++) {
//					unsigned char val = bufAsChar[idx*3+2];
//					outBuf[idx] = val;
//				}
//				m_currentRGB->get(2)->updateTexture(outBuf.data(), false, range);
//			}
		}

		emit currentIndexChanged(m_currentImageIndex);
	}
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMulti::createDataFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, QObject *parent) {
	QString cubeFile;

	QStringList rgb2Paths;
	QStringList rgb2Names;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";

	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				QFileInfoList rgb2InfoList = dir.entryInfoList(QStringList() << "*", QDir::Dirs |
						QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
				for (const QFileInfo& rgb2Info : rgb2InfoList) {
					rgb2Paths << rgb2Info.absoluteFilePath();
					rgb2Names << rgb2Info.fileName();
				}
			}
		}
	}
	bool isValid = rgb2Names.count()>0;
	bool errorLogging = false;
	int rgb2Index = 0;
	if (isValid) {
		QStringList rgb2NamesBuf = rgb2Names;
		StringSelectorDialog dialog(&rgb2NamesBuf, "Select RGB2");
		int result = dialog.exec();
		rgb2Index = dialog.getSelectedIndex();

		isValid = result==QDialog::Accepted && rgb2Index<rgb2Names.count() && rgb2Index>=0;
	} else {
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB2 data");
		errorLogging = true;
	}

	FixedRGBLayersFromDatasetAndCube* outObj = nullptr;
	QString sismageName;
	if (isValid) {
		QDir dir = QFileInfo(rgb2Paths[rgb2Index]).dir();
		isValid = dir.cdUp();
		sismageName = dir.dirName();
	}
	if (isValid) {
		cubeFile = rgb2Paths[rgb2Index];
		QString cubeName = rgb2Names[rgb2Index];
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);

			bool isRgb1 = false;

			if (isValid) {
				outObj = new FixedRGBLayersFromDatasetAndCubeImplMulti(
							cubeFile, prefix+sismageName+" "+cubeName+" (rgb2)", isRgb1, workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMulti::createDataFromDatasetWithUIRgb1(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, QObject *parent) {
	QString cubeFile;

	QStringList rgb2Paths;
	QStringList rgb2Names;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";

	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				QFileInfoList rgb2InfoList = dir.entryInfoList(QStringList() << "*", QDir::Dirs |
						QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
				for (const QFileInfo& rgb2Info : rgb2InfoList) {
					rgb2Paths << rgb2Info.absoluteFilePath();
					rgb2Names << rgb2Info.fileName();
				}
			}
		}
	}
	bool isValid = rgb2Names.count()>0;
	bool errorLogging = false;
	int rgb2Index = 0;
	if (isValid) {
		QStringList rgb2NamesBuf = rgb2Names;
		StringSelectorDialog dialog(&rgb2NamesBuf, "Select RGB2");
		int result = dialog.exec();
		rgb2Index = dialog.getSelectedIndex();

		isValid = result==QDialog::Accepted && rgb2Index<rgb2Names.count() && rgb2Index>=0;
	} else {
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB2 data");
		errorLogging = true;
	}

	FixedRGBLayersFromDatasetAndCube* outObj = nullptr;
	QString sismageName;
	if (isValid) {
		QDir dir = QFileInfo(rgb2Paths[rgb2Index]).dir();
		isValid = dir.cdUp();
		sismageName = dir.dirName();
	}
	if (isValid) {
		cubeFile = rgb2Paths[rgb2Index];
		QString cubeName = rgb2Names[rgb2Index];
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);

			bool isRgb1 = true;

			if (isValid) {
				outObj = new FixedRGBLayersFromDatasetAndCubeImplMulti(
							cubeFile, prefix+sismageName+" "+cubeName+" (rgb1)", isRgb1, workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::dirPath() const {
	return m_dirPath;
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::surveyPath() const {
	return QString::fromStdString(SismageDBManager::rgt2rgbPath2SurveyPath(dirPath().toStdString()));
}

template<typename T>
void FixedRGBLayersFromDatasetAndCubeImplMulti::CopyGDALBufToFloatBufInterleaved<T>::run(const void* _oriBuf,
		short* outBuf, std::size_t width, std::size_t height, std::size_t numBands,
		std::size_t offset, ImageFormats::QColorFormat colorFormat, GDALRasterBand* hBand) {
	const T* oriBuf = static_cast<const T*>(_oriBuf);
	std::size_t N = width*height;
	if (colorFormat==ImageFormats::QColorFormat::GRAY) {
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[i*3+c] = oriBuf[i];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_INDEXED) {
		GDALColorTable * colorTable = hBand->GetColorTable();
		long colorTabelSize = colorTable->GetColorEntryCount();
		std::vector<short> colorTableBuf;
		colorTableBuf.resize(colorTabelSize*3);
		for (int colorTableIdx=0; colorTableIdx<colorTabelSize; colorTableIdx++) {
			const GDALColorEntry* entry = colorTable->GetColorEntry(colorTableIdx);
			colorTableBuf[colorTableIdx*3] = entry->c1;
			colorTableBuf[colorTableIdx*3+1] = entry->c2;
			colorTableBuf[colorTableIdx*3+2] = entry->c3;
		}
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			long idx = oriBuf[i]; // apply index
			if (idx>=0 && idx<colorTabelSize) {
				//const GDALColorEntry* entry = colorTable->GetColorEntry(idx);
				short r,g,b;
				if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_Gray) {
					r = colorTableBuf[idx*3]; //entry->c1;
					g = r;
					b = r;
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_RGB) {
					r = colorTableBuf[idx*3]; //entry->c1;
					g = colorTableBuf[idx*3+1]; //entry->c2;
					b = colorTableBuf[idx*3+2]; //entry->c3;
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_CMYK) {
					r = colorTableBuf[idx*3]; //entry->c1;
					g = colorTableBuf[idx*3+1]; //entry->c2;
					b = colorTableBuf[idx*3+2]; //entry->c3;
					qDebug() << "Unexpected color encoding : CMYK";
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_HLS) {
					r = colorTableBuf[idx*3]; //entry->c1;
					g = colorTableBuf[idx*3+1]; //entry->c2;
					b = colorTableBuf[idx*3+2]; //entry->c3;
					qDebug() << "Unexpected color encoding : HLS";
				}
				outBuf[i*3] = r;
				outBuf[i*3+1] = g;
				outBuf[i*3+2] = b;
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGB_INTERLEAVED) {
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[i*3+c] = oriBuf[i*3+c];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[i*3+c] = oriBuf[i*4+c];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGB_PLANAR) {
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[i*3+c] = oriBuf[c*N+i];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_PLANAR) {
#pragma omp parallel for
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[i*3+c] = oriBuf[c*N+i];
			}
		}
	}
}

// oriBuf and outBuf must be fo of same tyme and size
void FixedRGBLayersFromDatasetAndCubeImplMulti::swapWidthHeight(const void* _oriBuf, void* _outBuf,
		std::size_t oriWidth, std::size_t oriHeight, std::size_t typeSize) const {
	const char* oriBuf = static_cast<const char*>(_oriBuf);
	char* outBuf = static_cast<char*>(_outBuf);
	for (std::size_t i=0; i<oriWidth; i++) {
		for (std::size_t j=0; j<oriHeight; j++) {
			std::size_t indexOri = (i + j*oriWidth) * typeSize;
			std::size_t indexOut = (j + i*oriHeight) * typeSize;
			for (std::size_t k=0; k<typeSize; k++) {
				outBuf[indexOut+k] = oriBuf[indexOri+k];
			}
		}
	}
}

bool FixedRGBLayersFromDatasetAndCubeImplMulti::readRgb1(const QString& path, short* buf,
		long w, long h) const {
	if (path.isNull() || path.isEmpty()) {
		return false;
	}

	GDALDataset* poDataset = (GDALDataset*) GDALOpen(path.toStdString().c_str(),
			GA_ReadOnly);
	bool valid = poDataset != nullptr && poDataset->GetRasterXSize()==w &&
			poDataset->GetRasterYSize()==h;

	if (valid) {
		GDALRasterBand* hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
		ImageFormats::QColorFormat colorFormat = GDALLoader::getColorFormatType(poDataset);
		ImageFormats::QSampleType sampleType = GDALLoader::getSampleType(hBand);
		GDALDataType type = GDALGetRasterDataType(hBand);
		int offset = GDALGetDataTypeSizeBytes(type);
		int numBands = poDataset->GetRasterCount();

		GDALColorTable * colorTable = hBand->GetColorTable();

		std::vector<char> tmpBuf;
		tmpBuf.resize(w*h*numBands*offset);

		valid = poDataset->RasterIO(GF_Read, 0, 0, w, h, tmpBuf.data(), w, h,
						type, numBands, nullptr, numBands * offset,
						numBands * offset * w, offset)==CPLErr::CE_None;

		if (valid) {
//			std::vector<short> swapBuf;
//			buf.resize(w*h*3);
			SampleTypeBinder binder(sampleType);
			binder.bind<CopyGDALBufToFloatBufInterleaved>(tmpBuf.data(), buf, w, h,
					numBands, offset, colorFormat, hBand);

			//swapWidthHeight(swapBuf.data(), buf.data(), w, h, sampleType.byte_size());
		}

		GDALClose(poDataset);
	}
	return valid;
}

std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters>>
FixedRGBLayersFromDatasetAndCubeImplMulti::findPotentialDataRgb1(const QString& searchPath) {
	std::vector<std::shared_ptr<Parameters>> output;
	QDir searchDir(searchPath);
	QFileInfoList rgb2InfoList = searchDir.entryInfoList(QStringList() << "*", QDir::Dirs |
			QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	for (const QFileInfo& rgb2Info : rgb2InfoList) {
		QDir dir(rgb2Info.absoluteFilePath());
		QFileInfoList files = dir.entryInfoList(
				QStringList() << "iso_*.raw", QDir::Files | QDir::Readable);

		int foundRgb1 = 0;
		for (const QFileInfo& e : files) {
			QString strVal = e.baseName().split("_").last();

			bool ok;
			int val = strVal.toInt(&ok);
			if (ok) {
				ok = dir.exists("rgb1_"+strVal+".png");
				if (ok) {
					foundRgb1++;
				}
			}
		}

		if (files.size()>0 && foundRgb1==files.size()) {
			output.push_back(std::make_shared<Parameters>(rgb2Info.fileName(), rgb2Info.absoluteFilePath(), true));
		}
	}
	return output;
}

std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters>>
FixedRGBLayersFromDatasetAndCubeImplMulti::findPotentialDataRgb2(const QString& searchPath) {
	std::vector<std::shared_ptr<Parameters>> output;
	QDir searchDir(searchPath);
	QFileInfoList rgb2InfoList = searchDir.entryInfoList(QStringList() << "*", QDir::Dirs |
			QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	for (const QFileInfo& rgb2Info : rgb2InfoList) {
		QDir dir(rgb2Info.absoluteFilePath());
		QFileInfoList files = dir.entryInfoList(QStringList() << "iso_*.raw", QDir::Files | QDir::Readable);

		int foundRgb2 = 0;
		for (const QFileInfo& e : files) {
			QString strVal = e.baseName().split("_").last();

			bool ok;
			int val = strVal.toInt(&ok);
			if (ok) {
				ok = dir.exists("rgb2_"+strVal+".raw");
				if (ok) {
					foundRgb2++;
				}
			}
		}
		if (files.size()>0 && foundRgb2==files.size()) {
			output.push_back(std::make_shared<Parameters>(rgb2Info.fileName(), rgb2Info.absoluteFilePath(), false));
		}
	}
	return output;
}

FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters::Parameters(QString name, QString dirPath, bool rgb1Valid) :
	FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams(name, rgb1Valid) {
	m_dirPath = dirPath;
}

FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters::~Parameters() {

}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters::create(QString name,
				WorkingSetManager *workingSet, const Grid3DParameter& params,
				QObject *parent) {
	return new FixedRGBLayersFromDatasetAndCubeImplMulti(m_dirPath, name, rgb1Valid(), workingSet, params, parent);
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters::sismageName(bool* ok) const {
	QDir dir = QFileInfo(m_dirPath).dir();
	bool isValid = dir.cdUp();
	QString sismageName;
	if (isValid) {
		sismageName = dir.dirName();
	} else {
		sismageName = "Invalid data";
	}
	*ok = isValid;
	return sismageName;
}

QString FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters::dirPath() const {
	return m_dirPath;
}
