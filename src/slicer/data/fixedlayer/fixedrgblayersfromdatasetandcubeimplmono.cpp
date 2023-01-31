#include "fixedrgblayersfromdatasetandcubeimplmono.h"
#include "sismagedbmanager.h"
#include "seismicsurvey.h"
#include "smdataset3D.h"
#include "datasetrelatedstorageimpl.h"
#include "GeotimeProjectManagerWidget.h"
#include "stringselectordialog.h"
#include "Xt.h"

#include <gdal.h>
#include <memory>

#include <QDir>
#include <QMessageBox>

FixedRGBLayersFromDatasetAndCubeImplMono::FixedRGBLayersFromDatasetAndCubeImplMono(QString cube,
			QString name, WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent) : FixedRGBLayersFromDatasetAndCube(name, workingSet, params, parent) {
	loadIsoAndRgb(cube);

	initLayersList();

	setCurrentImageIndex(0);
}

FixedRGBLayersFromDatasetAndCubeImplMono::FixedRGBLayersFromDatasetAndCubeImplMono(QString rgb2Path, QString rgb1Path, QString name,
		WorkingSetManager *workingSet, const Grid3DParameter& params,
		QObject *parent) : FixedRGBLayersFromDatasetAndCube(name, workingSet, params, parent) {
	loadIsoAndRgb(rgb2Path, rgb1Path);

	initLayersList();

	setCurrentImageIndex(0);
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::getObjFile(int index) const
{
	QFileInfo fileinfo(m_rgb2Path);
	QString temp = fileinfo.completeBaseName();

	QDir dir = fileinfo.absoluteDir();
	if(!dir.exists(temp))
	{
		dir.mkdir(temp);
	}

	dir.cd(temp);
	QString res = QString::number(index*isoStep()+isoOrigin())+".obj";//3do

	return dir.absoluteFilePath(res);
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::rgb2Path() const {
	return m_rgb2Path;
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::rgb1Path() const {
	return m_rgb1Path;
}

FixedRGBLayersFromDatasetAndCubeImplMono::~FixedRGBLayersFromDatasetAndCubeImplMono() {
	if (m_fRgb2!=nullptr) {
		fclose(m_fRgb2);
	}
	if (m_fRgb1!=nullptr) {
		fclose(m_fRgb1);
	}
}

void FixedRGBLayersFromDatasetAndCubeImplMono::loadIsoAndRgb(QString rgb2, QString rgb1) {
	m_fRgb2 = fopen(rgb2.toStdString().c_str(), "r");
	m_rgb2Path = rgb2;
	std::size_t sz;
	bool isValid;
	if (m_fRgb2!=nullptr) {
		fseek(m_fRgb2, 0L, SEEK_END);
		sz = ftell(m_fRgb2);
		fseek(m_fRgb2, 0L, SEEK_SET);
		isValid = true;
	}

	if (isValid) {
		m_numLayers = sz / (width() * depth() * sizeof(short) * 4 );

		m_fRgb1 = fopen(rgb1.toStdString().c_str(), "r");
		if (m_fRgb1!=nullptr) {
			m_rgb1Path = rgb1;
			std::size_t sz1;
			fseek(m_fRgb1, 0L, SEEK_END);
			sz = ftell(m_fRgb1);
			fseek(m_fRgb1, 0L, SEEK_SET);
			m_useRgb1 = sz==m_numLayers*width() * depth() * sizeof(char) * 3;
			if (!m_useRgb1) {
				fclose(m_fRgb1);
			}
		} else {
			m_fRgb1 = nullptr;
			m_useRgb1 = false;
		}
	} else {
		m_numLayers = 0;
		m_fRgb1 = nullptr;
		m_useRgb1 = false;
	}
	m_isoOrigin = (m_numLayers-1) * (-m_isoStep);
}

void FixedRGBLayersFromDatasetAndCubeImplMono::getImageForIndex(long newIndex,
		CUDAImagePaletteHolder* redCudaBuffer, CUDAImagePaletteHolder* greenCudaBuffer,
		CUDAImagePaletteHolder* blueCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return;

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
		buf.resize(layerSize*4);
		outBuf.resize(layerSize);
		size_t absolutePosition = layerSize * newIndex * sizeof(short) * 4;
		{
			QMutexLocker b2(&m_lockRgb2);
			fseek(m_fRgb2, absolutePosition, SEEK_SET);
			fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);
		}

		for (std::size_t idx=0; idx<layerSize; idx++) {
			short val = buf[idx*4+3];
			//swap(val);
			outBuf[idx] = val;
		}
		isoCudaBuffer->updateTexture(outBuf.data(), false);

		if (!m_useRgb1) {
			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*4];
				// swap
				//swap(val);
				outBuf[idx] = val;
			}
			redCudaBuffer->updateTexture(outBuf.data(), false);

			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*4+1];
				//swap(val);
				outBuf[idx] = val;
			}
			greenCudaBuffer->updateTexture(outBuf.data(), false);

			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*4+2];
				//swap(val);
				outBuf[idx] = val;
			}
			blueCudaBuffer->updateTexture(outBuf.data(), false);
		} else {
			size_t absolutePositionRgb1 = layerSize * newIndex * sizeof(unsigned char) * 3;
			{
				QMutexLocker b1(&m_lockRgb1);
				fseek(m_fRgb1, absolutePositionRgb1, SEEK_SET);
				fread(buf.data(), sizeof(unsigned char), layerSize*3, m_fRgb1);
			}
            unsigned char* bufAsChar = static_cast<unsigned char*>(static_cast<void*>(buf.data()));

            for (std::size_t idx=0; idx<layerSize; idx++) {
                    unsigned char val = bufAsChar[idx*3];
                    outBuf[idx] = val;
            }
			redCudaBuffer->updateTexture(outBuf.data(), false);

            for (std::size_t idx=0; idx<layerSize; idx++) {
                    unsigned char val = bufAsChar[idx*3+1];
                    outBuf[idx] = val;
            }
			greenCudaBuffer->updateTexture(outBuf.data(), false);

            for (std::size_t idx=0; idx<layerSize; idx++) {
                    unsigned char val = bufAsChar[idx*3+2];
                    outBuf[idx] = val;
            }
			blueCudaBuffer->updateTexture(outBuf.data(), false);

		}
	}
}

template<typename InputType>
bool checkValidityVect(const std::vector<InputType>& vect, std::size_t expectedSize) {
	return vect.size()>0 && vect.size()==expectedSize && vect.data()!=nullptr;
}

bool FixedRGBLayersFromDatasetAndCubeImplMono::getImageForIndex(long newIndex,
		QByteArray& rgbBuffer, QByteArray& isoBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return false;

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
		buf.resize(layerSize*4);
		isValid = checkValidityVect(buf, layerSize*4);

		if (isValid) {
//			{
//				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
//				if (isoFile!=NULL) {
////					fseek(isoFile, absolutePosition, SEEK_SET);
//					fread(isoBuffer.data(), sizeof(short), layerSize, isoFile);
//					fclose(isoFile);
//				} else {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
//				}
//			}
//			if (m_useRgb1) {
//				// TODO
//				bool ok = readRgb1(getRgb1FileFromIndex(newIndex), static_cast<short*>(static_cast<void*>(rgbBuffer.data())), w, h);
//				if (!ok) {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB1";
//				}
//			} else {
//				FILE* rgb2File = fopen(getRgb2FileFromIndex(newIndex).toStdString().c_str(), "r");
//				if (rgb2File!=NULL) {
//					fread(rgbBuffer.data(), sizeof(short), layerSize*3, rgb2File);
//					fclose(rgb2File);
//				} else {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
//				}
//			}
			size_t absolutePosition = layerSize * newIndex * sizeof(short) * 4;
			{
				QMutexLocker b2(&m_lockRgb2);
				fseek(m_fRgb2, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);
			}
			if (!m_useRgb1) {
				short* rgbBufferAsShort = static_cast<short*>(static_cast<void*>(rgbBuffer.data()));
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = buf[idx*4];
					// swap
					//swap(val);
					rgbBufferAsShort[idx*3] = val;
				}

				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = buf[idx*4+1];
					//swap(val);
					rgbBufferAsShort[idx*3+1] = val;
				}

				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = buf[idx*4+2];
					//swap(val);
					rgbBufferAsShort[idx*3+2] = val;
				}
			}

			short* isoBufferAsShort = static_cast<short*>(static_cast<void*>(isoBuffer.data()));
			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*4+3];
				//swap(val);
				isoBufferAsShort[idx] = val;
			}

			if (m_useRgb1) {
				size_t absolutePositionRgb1 = layerSize * newIndex * sizeof(char) * 3;
				{
					QMutexLocker b1(&m_lockRgb1);
					fseek(m_fRgb1, absolutePositionRgb1, SEEK_SET);
					fread(buf.data(), sizeof(unsigned char), layerSize*3, m_fRgb1); // reuse buffer
				}
				unsigned char* bufAsChar = static_cast<unsigned char*>(static_cast<void*>(buf.data()));
				short* rgbBufferAsShort = static_cast<short*>(static_cast<void*>(rgbBuffer.data()));

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3];
					rgbBufferAsShort[idx*3+1] = val;
				}

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3+1];
					rgbBufferAsShort[idx*3+1] = val;
				}

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3+2];
					rgbBufferAsShort[idx*3+2] = val;
				}
			}
		}
	}
	return isValid;
}

void FixedRGBLayersFromDatasetAndCubeImplMono::setCurrentImageIndexInternal(long newIndex) {
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

			std::vector<short> rgb2Buf;
			rgb2Buf.resize(layerSize*4);
//			outBuf.resize(layerSize);
			//resize = std::chrono::steady_clock::now();
//			{
//				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
//				if (isoFile!=NULL) {
//				//	initRead = std::chrono::steady_clock::now();
//					fread(grayBuf.data(), sizeof(short), layerSize, isoFile);
//					fclose(isoFile);
//				//	isoRead = std::chrono::steady_clock::now();
//					m_currentIso->updateTexture(grayBuf, false);
//					//isoTexture = std::chrono::steady_clock::now();
//				} else {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
//				}
//			}
//			//rgb1Read = isoTexture;
//			if (m_useRgb1) {
//				// TODO
//				short* bufShort = static_cast<short*>(static_cast<void*>(buf.data()));
//				bool ok = readRgb1(getRgb1FileFromIndex(newIndex), bufShort, w, h);
//				//rgb1Read = std::chrono::steady_clock::now();
//				if (ok) {
//					QVector2D range(0, 255);
//					m_currentRGB->updateTexture(buf, false, range, range, range);
//				} else {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB1";
//				}
//			} else {
//				FILE* rgb2File = fopen(getRgb2FileFromIndex(newIndex).toStdString().c_str(), "r");
//				if (rgb2File!=NULL) {
//	//				fseek(rgb2File, absolutePosition, SEEK_SET);
//					fread(buf.data(), sizeof(short), layerSize*3, rgb2File);
//					fclose(rgb2File);
//					m_currentRGB->updateTexture(buf, false);
//				} else {
//					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
//				}
//			}
		//	auto end = std::chrono::steady_clock::now();

		/*	qDebug() << "All : " << std::chrono::duration<double, std::milli>(end-start).count() <<
					", Resize : " << std::chrono::duration<double, std::milli>(resize-start).count() <<
					", Fopen : " << std::chrono::duration<double, std::milli>(initRead-resize).count() <<
					", Iso Read : " << std::chrono::duration<double, std::milli>(isoRead-initRead).count() <<
					", Iso update : " << std::chrono::duration<double, std::milli>(isoTexture-isoRead).count() <<
					", Rgb1 read : " << std::chrono::duration<double, std::milli>(rgb1Read-isoTexture).count() <<
					", Rgb1 update : "<< std::chrono::duration<double, std::milli>(end - rgb1Read).count();*/

			size_t absolutePosition = layerSize * m_currentImageIndex * sizeof(short) * 4;
			{
				QMutexLocker b2(&m_lockRgb2);
				fseek(m_fRgb2, absolutePosition, SEEK_SET);
				fread(rgb2Buf.data(), sizeof(short), layerSize*4, m_fRgb2);
			}

			if (!m_useRgb1) {
				short* rgbBuf = static_cast<short*>(static_cast<void*>(buf.data()));
				//#pragma omp parallel for
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = rgb2Buf[idx*4];
					// swap
					//swap(val);
					rgbBuf[idx*3] = val;
				}
				//m_currentRGB->get(0)->updateTexture(outBuf.data(), false);
				//#pragma omp parallel for
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = rgb2Buf[idx*4+1];
					//swap(val);
					rgbBuf[idx*3+1] = val;
				}
				//m_currentRGB->get(1)->updateTexture(outBuf.data(), false);
				//#pragma omp parallel for
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = rgb2Buf[idx*4+2];
					//swap(val);
					rgbBuf[idx*3+2] = val;
				}
				//m_currentRGB->get(2)->updateTexture(outBuf.data(), false);
				m_currentRGB->updateTexture(buf, false);
			}
			//#pragma omp parallel for
			short* grayBufAsShort = static_cast<short*>(static_cast<void*>(grayBuf.data()));
			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = rgb2Buf[idx*4+3];
				//swap(val);
				grayBufAsShort[idx] = val;
			}
			m_currentIso->updateTexture(grayBuf, false);

			if (m_useRgb1) {
				size_t absolutePositionRgb1 = layerSize * newIndex * sizeof(char) * 3;
				{
					QMutexLocker b1(&m_lockRgb1);
					fseek(m_fRgb1, absolutePositionRgb1, SEEK_SET);
					fread(rgb2Buf.data(), sizeof(unsigned char), layerSize*3, m_fRgb1); // reuse buffer
				}
				const unsigned char* bufAsChar = static_cast<const unsigned char*>(static_cast<const void*>(rgb2Buf.data()));
				short* rgbBuf = static_cast<short*>(static_cast<void*>(buf.data()));

				QVector2D range(0, 255);

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3];
					rgbBuf[idx*3] = val;
				}
//				m_currentRGB->get(0)->updateTexture(outBuf.data(), false, range);

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3+1];
					rgbBuf[idx*3+1] = val;
				}
//				m_currentRGB->get(1)->updateTexture(outBuf.data(), false, range);

				for (std::size_t idx=0; idx<layerSize; idx++) {
					unsigned char val = bufAsChar[idx*3+2];
					rgbBuf[idx*3+2] = val;
				}
//				m_currentRGB->get(2)->updateTexture(outBuf.data(), false, range);
				m_currentRGB->updateTexture(buf, false, range, range, range);
			}
		}

		emit currentIndexChanged(m_currentImageIndex);
	}
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::surveyPath() const {
	return QString::fromStdString(SismageDBManager::rgt2rgbPath2SurveyPath(rgb2Path().toStdString()));
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMono::createDataFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey,
		QObject *parent) {
	QString cube2File;

	QStringList rgb2Paths;
	QStringList rgb2Names;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";

	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				QFileInfoList rgb2InfoList = dir.entryInfoList(QStringList() << "rgb2_*.raw", QDir::Files | QDir::Readable);
				for (const QFileInfo& rgb2Info : rgb2InfoList) {
					rgb2Names << rgb2Info.fileName();
					rgb2Paths << rgb2Info.absoluteFilePath();
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
		cube2File = rgb2Paths[rgb2Index];
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);

			if (isValid) {
				outObj = new FixedRGBLayersFromDatasetAndCubeImplMono(
							cube2File, prefix+sismageName+" (rgb2)", workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMono::createDataFromDatasetWithUIRgb1(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey,
		QObject *parent) {
	QString cube1File;
	QString cube2File;

	QStringList rgb2Paths;
	QStringList rgb1Paths;
	QStringList rgb1Names;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";

	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				QFileInfoList rgb1InfoList = dir.entryInfoList(QStringList() << "rgb1_*raw" << "rgb1_*.rgb", QDir::Files | QDir::Readable);
				for (const QFileInfo& rgb1Info : rgb1InfoList) {
					QStringList removeAlpha = rgb1Info.fileName().split("__alpha_");
					QString rgb2NameFilterAlpha = extractListAndJoin(removeAlpha, 0, removeAlpha.count()-2, "__alpha_");
					QStringList removeFrom = rgb2NameFilterAlpha.split("_from_");
					QString rgb2Name;
					QString rgb2Path;

					QDir cubeRgt2RgbDir = rgb1Info.dir();
					bool found = false;
					long indexStrList = 1;
					QString rgb2NameFilterFrom;
					while (!found && indexStrList<removeFrom.count()) {
						rgb2NameFilterFrom = extractListAndJoin(removeFrom, indexStrList, removeFrom.count()-1, "_from_");
						rgb2Name = rgb2NameFilterFrom + ".raw";
						rgb2Path = cubeRgt2RgbDir.absoluteFilePath(rgb2Name);
						found = QFileInfo(rgb2Path).exists();
						if (!found)  {
							indexStrList++;
						}
					}
					if (found) {
						rgb1Paths << rgb1Info.absoluteFilePath();
						rgb1Names << rgb1Info.fileName();
						rgb2Paths << rgb2Path;
					}
				}
			}
		}
	}
	bool isValid = rgb1Names.count()>0;
	bool errorLogging = false;
	int rgb2Index = 0;
	if (isValid) {
		QStringList rgb1NamesBuf = rgb1Names;
		StringSelectorDialog dialog(&rgb1NamesBuf, "Select RGB1");
		int result = dialog.exec();
		rgb2Index = dialog.getSelectedIndex();

		isValid = result==QDialog::Accepted && rgb2Index<rgb1Names.count() && rgb2Index>=0;
	} else {
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB1 data");
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
		cube1File = rgb1Paths[rgb2Index];
		cube2File = rgb2Paths[rgb2Index];
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);

			if (isValid) {
				outObj = new FixedRGBLayersFromDatasetAndCubeImplMono(
							cube2File, cube1File, prefix+sismageName+" (rgb1)", workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMono::Parameters>>
FixedRGBLayersFromDatasetAndCubeImplMono::findPotentialDataRgb1(const QString& searchPath) {
	std::vector<std::shared_ptr<Parameters>> output;
	QDir dir(searchPath);
	QFileInfoList rgb1InfoList = dir.entryInfoList(QStringList() << "rgb1_*raw" << "rgb1_*.rgb", QDir::Files | QDir::Readable);
	for (const QFileInfo& rgb1Info : rgb1InfoList) {
		QStringList removeAlpha = rgb1Info.fileName().split("__alpha_");
		QString rgb2NameFilterAlpha = extractListAndJoin(removeAlpha, 0, removeAlpha.count()-2, "__alpha_");
		QStringList removeFrom = rgb2NameFilterAlpha.split("_from_");
		QString rgb2Name;
		QString rgb2Path;

		QDir cubeRgt2RgbDir = rgb1Info.dir();
		bool found = false;
		long indexStrList = 1;
		QString rgb2NameFilterFrom;
		while (!found && indexStrList<removeFrom.count()) {
			rgb2NameFilterFrom = extractListAndJoin(removeFrom, indexStrList, removeFrom.count()-1, "_from_");
			rgb2Name = rgb2NameFilterFrom + ".raw";
			rgb2Path = cubeRgt2RgbDir.absoluteFilePath(rgb2Name);
			found = QFileInfo(rgb2Path).exists();
			if (!found)  {
				indexStrList++;
			}
		}
		if (found) {
			output.push_back(std::make_shared<Parameters>(rgb1Info.fileName(), rgb2Path, rgb1Info.absoluteFilePath()));
		}
	}
	return output;
}

std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMono::Parameters>>
FixedRGBLayersFromDatasetAndCubeImplMono::findPotentialDataRgb2(const QString& searchPath) {
	std::vector<std::shared_ptr<Parameters>> output;
	QDir dir(searchPath);
	QFileInfoList rgb2InfoList = dir.entryInfoList(QStringList() << "rgb2_*.raw", QDir::Files | QDir::Readable);
	for (const QFileInfo& rgb2Info : rgb2InfoList) {
		output.push_back(std::make_shared<Parameters>(rgb2Info.fileName(), rgb2Info.absoluteFilePath(), ""));
	}
	return output;
}

FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::Parameters(QString name, QString rgb2Path, QString rgb1Path) :
	FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams(name, !(rgb1Path.isNull() || rgb1Path.isEmpty())) {
	m_rgb2Path = rgb2Path;
	m_rgb1Path = rgb1Path;
}

FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::~Parameters() {

}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::create(QString name,
				WorkingSetManager *workingSet, const Grid3DParameter& params,
				QObject *parent) {
	if (rgb1Valid()) {
		return new FixedRGBLayersFromDatasetAndCubeImplMono(m_rgb2Path, m_rgb1Path, name, workingSet, params, parent);
	} else {
		return new FixedRGBLayersFromDatasetAndCubeImplMono(m_rgb2Path, name, workingSet, params, parent);
	}
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::sismageName(bool* ok) const {
	QDir dir = QFileInfo(m_rgb2Path).dir();
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

QString FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::rgb2Path() const {
	return m_rgb2Path;
}

QString FixedRGBLayersFromDatasetAndCubeImplMono::Parameters::rgb1Path() const {
	return m_rgb1Path;
}
