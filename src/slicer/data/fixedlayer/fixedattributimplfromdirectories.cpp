#include "fixedattributimplfromdirectories.h"

#include "sismagedbmanager.h"
#include "smdataset3D.h"
#include "seismicsurvey.h"
#include "datasetrelatedstorageimpl.h"
#include "stringselectordialog.h"
#include "GeotimeProjectManagerWidget.h"
#include "sampletypebinder.h"
#include "gdalloader.h"
#include <freeHorizonManager.h>
#include <freeHorizonQManager.h>
#include <rgtSpectrumHeader.h>
// #include "Xt.h"

#include "gdal.h"
#include <gdal_priv.h>
#include <memory>

#include <QDir>
#include <QMessageBox>


FixedAttributImplFromDirectories::FixedAttributImplFromDirectories(QString dirPath, QString dirName, QString seismicName, WorkingSetManager *workingSet,
		const Grid3DParameter& params, QObject *parent) : FixedRGBLayersFromDatasetAndCube(dirName, workingSet, params, parent) {
	// m_dataType = dataType;
//	loadIsoAndRgb(rgb2Path, rgb1Path);

	m_useRgb1 = false;
	m_dirPath = dirPath;
	m_dirName = dirName;
	m_seismicName = seismicName;

	loadObjectParamsFromDir(dirPath, dirName, seismicName);
	initLayersList();
	setCurrentImageIndex(0);
}

QColor FixedAttributImplFromDirectories::getHorizonColor()
{
	bool ok = false;
	QColor col = FreeHorizonQManager::loadColorFromPath(m_dirPath, &ok);
	if ( ok ) return col;
	return QColor(Qt::white);
}

/*
GraphEditor_PolyLineShape *FixedAttributImplFreeHorizonFromDirectories::getHorizonShape()
{
	return m_polylineShape;
}
*/

void FixedAttributImplFromDirectories::setHorizonColor(QColor col)
{
	FreeHorizonQManager::saveColorToPath(m_dirPath, col);
	emit colorChanged(col);
}


QString FixedAttributImplFromDirectories::getObjFile(int index) const
{
	return "";
}

QString FixedAttributImplFromDirectories::getIsoFileFromIndex(int index) {
	return m_isoNames[index];
}

QString FixedAttributImplFromDirectories::getSpectrumFileFromIndex(int index) const {
	return m_spectrumNames[index];
}

QString FixedAttributImplFromDirectories::getGccFileFromIndex(int index) const {
	return m_gccNames[index];
}

QString FixedAttributImplFromDirectories::getMeanFileFromIndex(int index) const {
	return m_meanNames[index];
}

QString FixedAttributImplFromDirectories::getAttributFileFromIndex(int index) {
	// QComboBox *cb = static_cast<QComboBox*>(getOption());
	// if ( cb == nullptr ) return "";
	// int idx = getOptionAttribut();
	// if ( idx == 0 ) return getIsoFileFromIndex(index);
	// if ( idx == 1 ) return getSpectrumFileFromIndex(index);
	// if ( idx == 2 ) return getGccFileFromIndex(index);
	// if ( idx == 3 ) return getMeanFileFromIndex(index);
	return m_attributNames[index];
}

QString FixedAttributImplFromDirectories::readAttributFromFile(int index, void *buff, long size)
{
	// QComboBox *cb = static_cast<QComboBox*>(getOption());
	// if ( cb == nullptr ) return "error";
	int idx = getOptionAttribut();
	QString filename = getAttributFileFromIndex(index);
	// QString attributBaseName = m_dirName.section('_', -1);
	QString attributBaseName =  QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));


	if ( attributBaseName == ISODATA_NAME ||  attributBaseName == ISODATA_OLDNAME )
	{
		short *tmp = (short *)calloc(size, sizeof(short));
		if ( tmp != nullptr )
		{
			FreeHorizonManager::readInt32(filename.toStdString(), (short*)tmp);
			for (long n=0; n<size; n++)
			{
				((short*)buff)[3*n] = tmp[n];
				((short*)buff)[3*n+1] = tmp[n];
				((short*)buff)[3*n+2] = tmp[n];
			}
			free(tmp);
		}
	}
	if ( attributBaseName == "spectrum" || attributBaseName == "gcc" )
	{
		FreeHorizonManager::attributRead(filename.toStdString(), buff);
	}
	else if ( attributBaseName == "mean" )
	{
		short *tmp = (short *)calloc(size, sizeof(short));
		FreeHorizonManager::attributRead(filename.toStdString(), tmp);
		for (long n=0; n<size; n++)
		{
			((short*)buff)[3*n] = tmp[n];
			((short*)buff)[3*n+1] = tmp[n];
			((short*)buff)[3*n+2] = tmp[n];
		}
		free(tmp);
	}
	return "ok";
}


FixedAttributImplFromDirectories::~FixedAttributImplFromDirectories() {

}

void FixedAttributImplFromDirectories::loadObjectParamsFromDir(const QString& dirPath, const QString& dirName, const QString& seismicName)
{
	QDir mainDir(dirPath);
	QFileInfoList mainDirList = mainDir.entryInfoList(QStringList() << "iso_*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	int N = mainDirList.size();
	if ( N > 0 )
	{
		m_attributNames.clear();
		m_attributNames.resize(N);
		m_isoNames.clear();
		m_isoNames.resize(N);

		for (int i=0; i<N; i++)
		{
			QString attributName;
			if ( m_dirName == ISODATA_NAME )
			{
				attributName = m_dirName + ".iso";
			}
			else
			{
				attributName = m_dirName + QString::fromStdString(FreeHorizonManager::attributExt);
				QString filePath = mainDirList[i].absoluteFilePath() + "/" + attributName;
				if ( !FreeHorizonManager::exist(filePath.toStdString() ) )
				{
					attributName = m_dirName + ".raw";
				}
			}
			m_attributNames[i] = mainDirList[i].absoluteFilePath() + "/" + attributName;
			m_isoNames[i] = mainDirList[i].absoluteFilePath() + "/" + ISODATA_NAME + ".iso";
		}
		m_numLayers = N;
		m_isoOrigin = N;
		m_isoStep = -1;
	}
}

void FixedAttributImplFromDirectories::getImageForIndex(long newIndex,
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

		std::vector<float> bufF;
		std::vector<float> outBufF;
		bufF.resize(layerSize);
		outBufF.resize(layerSize);

		//size_t absolutePosition = layerSize * newIndex * sizeof(short) * 4;
		{
//			QMutexLocker b2(&m_lockRgb2);
//			fseek(m_fRgb2, absolutePosition, SEEK_SET);
//			fread(buf.data(), sizeof(short), layerSize*4, m_fRgb2);


			// std::string ret = FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)buf.data());
			std::string ret = FreeHorizonManager::read(getIsoFileFromIndex(newIndex).toStdString(), (short*)bufF.data());
			if ( ret == "ok" )
			{
				isoCudaBuffer->updateTexture(bufF.data(), false);
			}
			else
			{
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
			}
		}

		bool rgbValid = false;
		QString ret = readAttributFromFile(newIndex, buf.data(), layerSize);
		if ( ret.compare("ok") == 0 )
		{
			rgbValid = true;
		}
		else
		{
			qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
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

bool FixedAttributImplFromDirectories::getImageForIndex(long newIndex,
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
	isoBuffer.resize(layerSize* sizeof(float));

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
				// std::string ret = FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)isoBuffer.data());
				std::string ret = FreeHorizonManager::read(getIsoFileFromIndex(newIndex).toStdString(), (float*)isoBuffer.data());
				if ( ret != "ok" ) qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
			}
			QString ret = readAttributFromFile(newIndex, rgbBuffer.data(), layerSize);
			if ( ret.compare("ok") != 0 ) qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
		}
	}
	return isValid;
}

void FixedAttributImplFromDirectories::setCurrentImageIndexInternal(long newIndex) {
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
				std::string ret = FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)grayBuf.data());
				if ( ret == "ok" )
				{
					m_currentIso->updateTexture(grayBuf, false);
				}
				else
				{
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
				}

				/*
				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if (isoFile!=NULL) {
					//	initRead = std::chrono::steady_clock::now();
					float *tmp = (float*)calloc(layerSize, sizeof(float));
					short *tmp2 = (short*)calloc(layerSize, sizeof(short));
					if ( tmp != nullptr && tmp2 != nullptr ){
						fread(tmp, sizeof(float), layerSize, isoFile);
						for (long i=0; i<layerSize; i++) tmp2[i] = (short)tmp[i];
						memcpy(grayBuf.data(), tmp2, layerSize*sizeof(short));
						free(tmp);
						free(tmp2);
						fclose(isoFile);
						//	isoRead = std::chrono::steady_clock::now();
						m_currentIso->updateTexture(grayBuf, false);
					}
					//isoTexture = std::chrono::steady_clock::now();
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
				}
				*/
			}
			//rgb1Read = isoTexture;

			/*
			FILE* rgb2File = fopen(getSpectrumFileFromIndex(newIndex).toStdString().c_str(), "r");
			if (rgb2File!=NULL) {
				//				fseek(rgb2File, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize*3, rgb2File);
				fclose(rgb2File);
				m_currentRGB->updateTexture(buf, false);
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
			}
			*/
			QString ret = readAttributFromFile(newIndex, buf.data(), layerSize);
			if ( ret.compare("ok") == 0 )
			{
				m_currentRGB->updateTexture(buf, false);
			}
			else
			{
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
			}
		//	auto end = std::chrono::steady_clock::now();

		/*	qDebug() << "All : " << std::chrono::duration<double, std::milli>(end-start).count() <<
					", Resize : " << std::chrono::duration<double, std::milli>(resize-start).count() <<
					", Fopen : " << std::chrono::duration<double, std::milli>(initRead-resize).count() <<
					", Iso Read : " << std::chrono::duration<double, std::milli>(isoRead-initRead).count() <<
					", Iso update : " << std::chrono::duration<double, std::milli>(isoTexture-isoRead).count() <<
					", Rgb1 read : " << std::chrono::duration<double, std::milli>(rgb1Read-isoTexture).count() <<
					", Rgb1 update : "<< std::chrono::duration<double, std::milli>(end - rgb1Read).count();*/

		}
		emit currentIndexChanged(m_currentImageIndex);
	}
}

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFromDirectories::createDataFromDatasetWithUI(QString prefix,
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
				// outObj = new FixedAttributImplFromDirectories(cubeFile, prefix+sismageName+" "+cubeName+" (rgb2)", isRgb1, workingSet, params, "", parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFromDirectories::createDataFromDatasetWithUIRgb1(QString prefix,
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
				// outObj = new FixedAttributImplFromDirectories(cubeFile, prefix+sismageName+" "+cubeName+" (rgb1)", isRgb1, workingSet, params, "", parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

QString FixedAttributImplFromDirectories::dirPath() const {
	return m_dirPath;
}

QString FixedAttributImplFromDirectories::surveyPath() const {
	return QString::fromStdString(SismageDBManager::rgt2rgbPath2SurveyPath(dirPath().toStdString()));
}

template<typename T>
void FixedAttributImplFromDirectories::CopyGDALBufToFloatBufInterleaved<T>::run(const void* _oriBuf,
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
void FixedAttributImplFromDirectories::swapWidthHeight(const void* _oriBuf, void* _outBuf,
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

bool FixedAttributImplFromDirectories::readRgb1(const QString& path, short* buf,
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



// ================================================================================================

QString FixedAttributImplFromDirectories::getSeismicNameFromFile(QString filename)
{
	int pos = filename.lastIndexOf(QChar('_'));
	return filename.left(pos);
}


std::vector<std::shared_ptr<FixedAttributImplFromDirectories::Parameters>>
FixedAttributImplFromDirectories::findPotentialData(const QString& searchPath)
{
	std::vector<QString> attibutList {"*_mean.raw", "*_spectrum.raw", "*_gcc.raw" };
	std::vector<std::shared_ptr<Parameters>> output;

	QDir mainDir(searchPath);
	if ( !mainDir.exists() ) return output;
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( mainList.size() == 0 ) return output;
	for (int i=0; i<mainList.size(); i++)
	{
		QDir dir0(mainList[i].absoluteFilePath());
		QFileInfoList list0 = dir0.entryInfoList(QStringList() << "iso_*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
		if ( list0.size() == 0 ) continue;

		qDebug() << list0[0].absoluteFilePath();
		QDir dir1(list0[0].absoluteFilePath());
		std::vector<QString> stringList;

		for (int k=0; k<attibutList.size(); k++)
		{
			QFileInfoList list1 = dir1.entryInfoList(QStringList() << attibutList[k], QDir::Files);
			for (int j=0; j<list1.size(); j++)
			{
				QString seismicName = getSeismicNameFromFile(list1[j].fileName());
				stringList.push_back(seismicName);
			}
		}
		std::sort( stringList.begin(), stringList.end() );
		stringList.erase( std::unique( stringList.begin(), stringList.end() ), stringList.end() );


		QString rgtName = mainList[i].fileName();
		for (int j=0; j<stringList.size(); j++)
		{
			QString seismicName = stringList[j];
			QString displayName = mainList[i].fileName() + "  [ on "  + seismicName + " ]";
			output.push_back(std::make_shared<Parameters>(displayName, mainList[i].absoluteFilePath(), true, "", seismicName));
		}
		// for (QString str:stringList ) qDebug() << str;
	}
	return output;
}










std::vector<std::shared_ptr<FixedAttributImplFromDirectories::Parameters>>
FixedAttributImplFromDirectories::findPotentialDataRgb1(const QString& searchPath) {
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
			output.push_back(std::make_shared<Parameters>(rgb2Info.fileName(), rgb2Info.absoluteFilePath(), true, "spectrum", ""));
		}
	}
	return output;
}

std::vector<std::shared_ptr<FixedAttributImplFromDirectories::Parameters>>
FixedAttributImplFromDirectories::findPotentialDataRgb2(const QString& searchPath, const QString &dataType) {
	std::vector<std::shared_ptr<Parameters>> output;
	// qDebug() << "***** " << searchPath;
	QDir searchDir(searchPath);
	QFileInfoList rgb2InfoList = searchDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	for (const QFileInfo& rgb2Info : rgb2InfoList) {
		QDir dir(rgb2Info.absoluteFilePath());
		QDir searchDir2(dir.path());
		QFileInfoList rgb2InfoList2 = searchDir2.entryInfoList(QStringList() << "iso_*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
		if ( rgb2InfoList2.size() > 0 )
		{
			// qDebug() << dir.dirName() << "------ " << dir.path() << "-------";
			output.push_back(std::make_shared<Parameters>(dir.dirName(), dir.path(), false, dataType, ""));
		}
	}
	return output;
}

FixedAttributImplFromDirectories::Parameters::Parameters(QString name, QString dirPath, bool rgb1Valid, QString dataType, QString seismicName) :
	FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams(name, rgb1Valid) {

	m_dirPath = dirPath;
	m_dirName = name;
	m_dataType = dataType;
	m_seismicName = seismicName;
}

FixedAttributImplFromDirectories::Parameters::~Parameters() {

}

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFromDirectories::Parameters::create(QString name,
				WorkingSetManager *workingSet, const Grid3DParameter& params,
				QObject *parent) {

	// qDebug() << m_dirPath;
	// qDebug() << name;
	// return new FixedAttributImplFromDirectories(m_dirPath, name, rgb1Valid(), workingSet, params, this->m_dataType, parent);

	// FixedRGBLayersFromDatasetAndCube *obj;
	// obj = new FixedAttributImplFromDirectories(m_dirPath, m_dirName, m_seismicName, workingSet, params, parent);
	return new FixedAttributImplFromDirectories(m_dirPath, m_dirName, m_seismicName, workingSet, params, parent);
}

QString FixedAttributImplFromDirectories::Parameters::sismageName(bool* ok) const {
	// qDebug() << m_dirPath;
	// QString sismageName = QFileInfo(m_dirPath).fileName();
	*ok = true;
	// QString path = surveyPath();
	// qDebug() << path;
	// return m_seismicName;
	return QString::fromStdString(FreeHorizonManager::dataSetNameWithPrefixGet(m_dirPath.toStdString()+"/iso_00000/"));
}

QString FixedAttributImplFromDirectories::Parameters::dirPath() const {
	return m_dirPath;
}

void FixedAttributImplFromDirectories::updateDataAttribut()
{
	int newIndex = m_currentImageIndex;
	m_currentImageIndex = newIndex - 1;
	setCurrentImageIndexInternal(newIndex);
}

bool FixedAttributImplFromDirectories::isInlineXLineDisplay()
{
	//if ( m_dirName == ISODATA_NAME || m_dirName == ISODATA_OLDNAME ) return true;
	return true;
}


