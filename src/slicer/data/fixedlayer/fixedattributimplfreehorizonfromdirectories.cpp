#include "fixedattributimplfreehorizonfromdirectories.h"

#include <fixedlayerfromdataset.h>
#include "sismagedbmanager.h"
#include <seismic3ddataset.h>
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
#include <QColor>
#include <QToolTip>
#include <QMessageBox>
#include <fileInformationWidget.h>
#include <freeHorizonManager.h>
#include <freeHorizonQManager.h>
#include <rgtSpectrumHeader.h>

FixedAttributImplFreeHorizonFromDirectories::FixedAttributImplFreeHorizonFromDirectories(QString dirPath, QString dirName, std::vector<QString> seismicName, WorkingSetManager *workingSet,
		const Grid3DParameter& params, QObject *parent) : FixedRGBLayersFromDatasetAndCube(dirName, workingSet, params, parent) {
	// m_dataType = dataType;
//	loadIsoAndRgb(rgb2Path, rgb1Path);

	m_useRgb1 = false;
	m_dirPath = dirPath;
	m_dirName = dirName;
	m_workingSetManager = workingSet;
	// m_seismicName = seismicName;
	m_dataSetNames = seismicName;


	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	loadObjectParamsFromDir(dirPath, dirName, "");
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//qDebug() << "finish ::loadObjectParamsFromDir: " << std::chrono::duration<double, std::milli>(end-start).count();



	initLayersList();

	m_dataSetPath = getSpectrumDataSet();




	setCurrentImageIndex(0);



	// emit frequencyChanged();

}
/*
void FixedAttributImplFreeHorizonFromDirectories::actionMenuCreate()
{
	m_actionColor = new QAction(tr("color"), this);
	connect(m_actionColor, &QAction::triggered, this, &FixedAttributImplFreeHorizonFromDirectories::trt_changeColor);
	m_actionProperties = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("properties"), this);
	connect(m_actionProperties, &QAction::triggered, this, &FixedAttributImplFreeHorizonFromDirectories::trt_properties);
	m_actionLocation = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("location"), this);
	connect(m_actionLocation, &QAction::triggered, this, &FixedAttributImplFreeHorizonFromDirectories::trt_location);
}
*/


QString FixedAttributImplFreeHorizonFromDirectories::getSpectrumDataSet()
{
	QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));
	if ( type != "spectrum" ) return "";
	QString tmp = m_dirPath;
	tmp = tmp.replace("//", "/");
	QString dataSetName1 = "seismic3d." + QString::fromStdString(FreeHorizonManager::dataSetFromAttributName(m_dirName.toStdString())) + ".xt";
	QString dataSetName2 = QString::fromStdString(FreeHorizonManager::dataSetFromAttributName(m_dirName.toStdString())) + ".xt";

	QFileInfo file(m_dirPath);
	QDir dir0 = file.absoluteDir();
	QDir dir(dir0);
	dir.cdUp();
	dir.cdUp();
	dir.cdUp();
	dir.cdUp();
	QString name1 = dir.absolutePath() + "/DATA/SEISMIC/" + dataSetName1;
	QFile file1(name1);
	if ( file1.exists() ) return name1;
	QString name2 = dir.absolutePath() + "/DATA/SEISMIC/" + dataSetName2;
	QFile file2(name2);
	if ( file2.exists() ) return name2;

	return "";
}

QColor FixedAttributImplFreeHorizonFromDirectories::getHorizonColor()
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

void FixedAttributImplFreeHorizonFromDirectories::setHorizonColor(QColor col)
{
	FreeHorizonQManager::saveColorToPath(m_dirPath, col);
	emit colorChanged(col);
}

QString FixedAttributImplFreeHorizonFromDirectories::getObjFile(int index) const
{
	return "";
}

QString FixedAttributImplFreeHorizonFromDirectories::getIsoFileFromIndex(int index) {
	return m_isoNames;
}

QString FixedAttributImplFreeHorizonFromDirectories::getSpectrumFileFromIndex(int index) {
	// qDebug() << m_spectrumNames;
	// return m_spectrumNames;
	return m_dirPath + "/" + getDataSetName() + "_" + QString::fromStdString(SPECTRUM_NAME) + QString::fromStdString(FreeHorizonManager::attributExt);
}

QString FixedAttributImplFreeHorizonFromDirectories::getGccFileFromIndex(int index) {
	return  m_dirPath + "/" + getDataSetName() + "_" + QString::fromStdString(GCC_NAME) + QString::fromStdString(FreeHorizonManager::attributExt);
}

QString FixedAttributImplFreeHorizonFromDirectories::getMeanFileFromIndex(int index) {
	return m_dirPath + "/" + getDataSetName() + "_" + QString::fromStdString(MEAN_NAME) + QString::fromStdString(FreeHorizonManager::attributExt);
}

QString FixedAttributImplFreeHorizonFromDirectories::getAttributFileFromIndex(int index) {
	// int idx =  getOptionAttribut();
	// if ( idx == 0 ) return getIsoFileFromIndex(index);
	// if ( idx == 1 ) return getSpectrumFileFromIndex(index);
	// if ( idx == 2 ) return getGccFileFromIndex(index);
	// if ( idx == 3 ) return getMeanFileFromIndex(index);
	if ( m_dirName == ISODATA_NAME || m_dirName == ISODATA_OLDNAME)
		return m_dirPath + "/" + m_dirName + ".iso";
	// for the old format
	if ( m_dirPath.contains("ImportExport/IJK/HORIZONS") )
		return m_dirPath + "/" + m_dirName + ".raw";
	else
		return m_dirPath + "/" + m_dirName + QString::fromStdString(FreeHorizonManager::attributExt);
}


FixedAttributImplFreeHorizonFromDirectories::~FixedAttributImplFreeHorizonFromDirectories() {

}

/*
void FixedAttributImplFreeHorizonFromDirectories::setDataSet3D(Seismic3DAbstractDataset *val)
{
	m_dataSet3D = val;
}
*/


int FixedAttributImplFreeHorizonFromDirectories::getRedIndex()
{
	if(m_redSet ==false)
	{
		if(m_greenSet ==false) getGreenIndex();
		else if(m_attributName=="spectrum")
		{
			m_paramSpectrum.f1 = std::max(0, m_paramSpectrum.f2-2);
		}
		m_redSet=true;
	}

	return m_paramSpectrum.f1;
}

int FixedAttributImplFreeHorizonFromDirectories::getGreenIndex()
{
	if(m_greenSet ==false)
	{
		if(m_attributName=="spectrum")
		{
			m_paramSpectrum.f2 = FreeHorizonManager::spectrumGetOptimalFGreenIndex(getAttributFileFromIndex(0).toStdString());
			if(m_redSet == false){ m_paramSpectrum.f1 = std::max(0, m_paramSpectrum.f2-2); m_redSet =true;}
			if(m_blueSet == false){  m_paramSpectrum.f3 = std::min(m_paramSpectrum.f2+2, m_nbreSpectrumFreq-1); m_blueSet = true;}

		}
		m_greenSet =true;
	}
	return m_paramSpectrum.f2;
}

int FixedAttributImplFreeHorizonFromDirectories::getBlueIndex()
{
	if(m_blueSet ==false)
	{
		if(m_greenSet ==false) getGreenIndex();
		else if(m_attributName=="spectrum")
		{
			m_paramSpectrum.f3 = std::min(m_paramSpectrum.f2+2, m_nbreSpectrumFreq-1);
		}
		m_blueSet=true;
	}
	return m_paramSpectrum.f3;
}

void FixedAttributImplFreeHorizonFromDirectories::loadObjectParamsFromDir(const QString& dirPath, const QString& dirName, const QString& seismicName)
{


	// m_gccNames = dirPath + "/" + seismicName + "_" + QString::fromStdString(GCC_NAME) + ".raw";
	// m_isoNames = dirPath + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);
	// m_meanNames = dirPath + "/" + seismicName + "_" + QString::fromStdString(MEAN_NAME) + ".raw";
	// m_spectrumNames = dirPath + "/" + seismicName + "_" + QString::fromStdString(SPECTRUM_NAME) + ".raw";

	m_gccNames = dirPath + "/" + QString::fromStdString(GCC_NAME) + "_" + seismicName  + QString::fromStdString(FreeHorizonManager::attributExt);
	m_isoNames = dirPath + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);
	m_meanNames = dirPath + "/" + QString::fromStdString(MEAN_NAME) + "_" + seismicName + QString::fromStdString(FreeHorizonManager::attributExt);
	m_spectrumNames = dirPath + "/" + QString::fromStdString(SPECTRUM_NAME) + "_" + seismicName + QString::fromStdString(FreeHorizonManager::attributExt);


	m_attributName = QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));
	if ( m_attributName == "spectrum" )
	{
		QString filename = getAttributFileFromIndex(0);

		m_nbreSpectrumFreq = FreeHorizonManager::getNbreSpectrumFreq(filename.toStdString());


	/*	std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
		m_paramSpectrum.f2 = FreeHorizonManager::spectrumGetOptimalFGreenIndex(filename.toStdString());
		std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
		qDebug() << "finish ::spectrumGetOptimalFGreenIndex : " << std::chrono::duration<double, std::milli>(end2-start2).count();

*/
	//	m_paramSpectrum.f1 = std::max(0, m_paramSpectrum.f2-2);
	//	m_paramSpectrum.f3 = std::min(m_paramSpectrum.f2+2, m_nbreSpectrumFreq-1);
	}
	m_numLayers = 1;
	m_isoOrigin = 0;
}


QString FixedAttributImplFreeHorizonFromDirectories::readAttributFromFile(int index, void *buff, long size)
{
	int idx =  getOptionAttribut();
	QString filename = getAttributFileFromIndex(index);

	// QString attributName = m_dirName.section('_', -1);
	QString attributName = QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));

	if ( m_dirName == ISODATA_NAME || m_dirName == ISODATA_OLDNAME )
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
	else if ( attributName == "spectrum"  )
	{
		FreeHorizonManager::readSpectrum(filename.toStdString(), getRedIndex(), getGreenIndex(), getBlueIndex(), (short*)buff);
	}
	else if ( attributName == "gcc" )
	{
		// FreeHorizonManager::readSpectrum(filename.toStdString(), m_paramGcc.f1, m_paramGcc.f2, m_paramGcc.f3, (short*)buff);
		short *tmp = (short*)calloc(size, sizeof(short));
		if ( tmp )
		{
			FreeHorizonManager::readGcc(filename.toStdString(), 0,  (short*)tmp);
			for (int n=0; n<size; n++)
			{
				((short*)buff)[3*n] = tmp[n];
				((short*)buff)[3*n+1] = tmp[n];
				((short*)buff)[3*n+2] = tmp[n];
			}
		}
		free(tmp);
	}
	else if ( attributName == "mean" )
	{
		FILE* pf = fopen(filename.toStdString().c_str(), "r");
		if ( pf != nullptr )
		{
			short *tmp = (short *)calloc(size, sizeof(short));
			fread(tmp, sizeof(short), size, pf);
			fclose(pf);
			for (long n=0; n<size; n++)
			{
				((short*)buff)[3*n] = tmp[n];
				((short*)buff)[3*n+1] = tmp[n];
				((short*)buff)[3*n+2] = tmp[n];
			}
			free(tmp);
		}
		else
		{
			memset(buff, 0, size*3*sizeof(short));
			return "error";
		}
	}
	// setColor(Qt::blue);
	return "ok";
}

void FixedAttributImplFreeHorizonFromDirectories::getImageForIndex(long newIndex,
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

			/*
			FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
			if (isoFile!=NULL) {
//				fseek(isoFile, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize, isoFile);
				fclose(isoFile);
				isoCudaBuffer->updateTexture(buf.data(), false);
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
			}
			*/

			//JD0
			// FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), buf.data());
			// isoCudaBuffer->updateTexture(buf.data(), false);
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

bool FixedAttributImplFreeHorizonFromDirectories::getImageForIndex(long newIndex,
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
				// JD0
				// FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)isoBuffer.data());

				/*
				FILE* isoFile = fopen(getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if (isoFile!=NULL) {
					fread(isoBuffer.data(), sizeof(short), layerSize, isoFile);
					fclose(isoFile);
				} else {
					qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read Iso";
				}
				*/
			}

			QString ret = readAttributFromFile(newIndex, rgbBuffer.data(), layerSize);
			if ( ret.compare("ok") != 0 ) qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";

			/*
			FILE* rgb2File = fopen(getSpectrumFileFromIndex(newIndex).toStdString().c_str(), "r");
			if (rgb2File!=NULL) {
				fread(rgbBuffer.data(), sizeof(short), layerSize*3, rgb2File);
				fclose(rgb2File);
			} else {
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
			}
			*/
		}
	}
	return isValid;
}

void FixedAttributImplFreeHorizonFromDirectories::setCurrentImageIndexInternal(long newIndex) {
	if (newIndex<0 || newIndex>=m_numLayers) {
		m_currentImageIndex = -1;
		return;
	}
	if (m_currentImageIndex==newIndex) {
		// return;
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
				FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)grayBuf.data());
				m_currentIso->updateTexture(grayBuf, false);

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

			QString ret = readAttributFromFile(newIndex, buf.data(), layerSize);
			if ( ret.compare("ok") == 0 )
			{
				m_currentRGB->updateTexture(buf, false);
			}
			else
			{
				qDebug() << "FixedRGBLayersFromDatasetAndCube : Failed to read RGB2";
			}
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

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFreeHorizonFromDirectories::createDataFromDatasetWithUI(QString prefix,
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
				// outObj = new FixedAttributImplFreeHorizonFromDirectories(cubeFile, prefix+sismageName+" "+cubeName+" (rgb2)", isRgb1, workingSet, params, "", parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFreeHorizonFromDirectories::createDataFromDatasetWithUIRgb1(QString prefix,
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
				// outObj = new FixedAttributImplFreeHorizonFromDirectories(cubeFile, prefix+sismageName+" "+cubeName+" (rgb1)", isRgb1, workingSet, params, "", parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

QString FixedAttributImplFreeHorizonFromDirectories::dirPath() const {
	return m_dirPath;
}

QString FixedAttributImplFreeHorizonFromDirectories::surveyPath() const {
	return QString::fromStdString(SismageDBManager::rgt2rgbPath2SurveyPath(dirPath().toStdString()));
}

template<typename T>
void FixedAttributImplFreeHorizonFromDirectories::CopyGDALBufToFloatBufInterleaved<T>::run(const void* _oriBuf,
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
void FixedAttributImplFreeHorizonFromDirectories::swapWidthHeight(const void* _oriBuf, void* _outBuf,
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

bool FixedAttributImplFreeHorizonFromDirectories::readRgb1(const QString& path, short* buf,
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
QString FixedAttributImplFreeHorizonFromDirectories::getSeismicNameFromFile(QString filename)
{
	int pos = filename.lastIndexOf(QChar('_'));
	return filename.left(pos);
}


std::vector<std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters>>
FixedAttributImplFreeHorizonFromDirectories::findPotentialData(const QString& searchPath)
{
	std::vector<std::shared_ptr<Parameters>> output;

	QDir mainDir(searchPath);
	if ( !mainDir.exists() ) return output;

	std::vector<QString> horizonNames = FreeHorizonQManager::getListName(searchPath);
	std::vector<QString> horizonPath = FreeHorizonQManager::getListPath(searchPath);
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( horizonNames.size() == 0 ) return output;
	for (int i=0; i<horizonPath.size(); i++)
	{
		std::vector<QString> dataSetNames = FreeHorizonQManager::getDataSet(horizonPath[i]);
		output.push_back(std::make_shared<Parameters>(horizonNames[i], horizonPath[i], true, "", dataSetNames));
	}
	return output;
}


std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters>
FixedAttributImplFreeHorizonFromDirectories::findData(const QString& horizonPath)
{
	std::shared_ptr<Parameters> output;

	QString horizonName = QString::fromStdString(FreeHorizonManager::getHorizonNameFromPath(horizonPath.toStdString()));
	std::vector<QString> dataSetNames = FreeHorizonQManager::getDataSet(horizonPath);
	output = std::make_shared<Parameters>(horizonName, horizonPath, true, "", dataSetNames);

	/*
	QDir mainDir(searchPath);
	if ( !mainDir.exists() ) return output;

	std::vector<QString> horizonNames = FreeHorizonQManager::getListName(searchPath);
	std::vector<QString> horizonPath = FreeHorizonQManager::getListPath(searchPath);
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( horizonNames.size() == 0 ) return output;
	for (int i=0; i<horizonPath.size(); i++)
	{
		std::vector<QString> dataSetNames = FreeHorizonQManager::getDataSet(horizonPath[i]);
		output.push_back(std::make_shared<Parameters>(horizonNames[i], horizonPath[i], true, "", dataSetNames));
	}
	*/
	return output;
}

FixedAttributImplFreeHorizonFromDirectories::Parameters::Parameters(QString name, QString dirPath, bool rgb1Valid, QString dataType, std::vector<QString> seismicName) :
	FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams(name, rgb1Valid) {
	m_dirPath = dirPath;
	m_dirName = name;
	m_dataType = dataType;
	m_dataSetNames = seismicName;
}

FixedAttributImplFreeHorizonFromDirectories::Parameters::~Parameters() {

}

FixedRGBLayersFromDatasetAndCube* FixedAttributImplFreeHorizonFromDirectories::Parameters::create(QString name,
				WorkingSetManager *workingSet, const Grid3DParameter& params,
				QObject *parent) {

	// qDebug() << m_dirPath;
	// qDebug() << name;
	// return new FixedAttributImplHorizonFromDirectories(m_dirPath, name, rgb1Valid(), workingSet, params, this->m_dataType, parent);
	return new FixedAttributImplFreeHorizonFromDirectories(m_dirPath, m_dirName, m_dataSetNames, workingSet, params, parent);
}

QString FixedAttributImplFreeHorizonFromDirectories::Parameters::sismageName(bool* ok) const {
	*ok = true;
	return QString::fromStdString(FreeHorizonManager::dataSetNameWithPrefixGet(m_dirPath.toStdString()));
}

QString FixedAttributImplFreeHorizonFromDirectories::Parameters::dirPath() const {
	return m_dirPath;
}


QString FixedAttributImplFreeHorizonFromDirectories::getPathUp(QString fullPath)
{
	QString out = fullPath;
	int pos = out.lastIndexOf("/");
	out = out.left(pos);
	// qDebug() << out;
	return out;
}

QString FixedAttributImplFreeHorizonFromDirectories::getFilenameFromPath(QString fullPath)
{
	QString out = fullPath;
	int pos = out.lastIndexOf("/");
	// out = out.left(pos);
	out = out.right(out.length()-pos-1);
	// qDebug() << out;
	return out;
}

QString FixedAttributImplFreeHorizonFromDirectories::getRgtFilename(QString fullPath)
{
	QString path0 = getPathUp(fullPath);
	path0 = getPathUp(path0);
	QString rgtFilename = getFilenameFromPath(path0);
	// qDebug() << rgtFilename;
	return rgtFilename;
}


// TODO
QString FixedAttributImplFreeHorizonFromDirectories::getSeismicFilename(QString fullPath)
{
	QString seismicFilename = "";
	QString filename = getFilenameFromPath(fullPath);
	int pos = filename.lastIndexOf("_isoData.raw");
	seismicFilename = filename.left(pos);
	return seismicFilename;
}

QString FixedAttributImplFreeHorizonFromDirectories::getHorizonName(QString fullPath)
{
	QString text = "freehorizon_";
	QString name = getPathUp(fullPath);
	name = getFilenameFromPath(name);
	int pos = name.lastIndexOf(text);
	name = name.right(name.length()-pos-text.length());
	return name;
}

void FixedAttributImplFreeHorizonFromDirectories::updateDataAttribut()
{
	int newIndex = m_currentImageIndex;
	m_currentImageIndex = newIndex - 1;
	setCurrentImageIndexInternal(newIndex);
}

bool FixedAttributImplFreeHorizonFromDirectories::isInlineXLineDisplay()
{
//	if ( m_dirName == ISODATA_NAME || m_dirName == ISODATA_OLDNAME) return true;
	// return false;
	return true;
}

void FixedAttributImplFreeHorizonFromDirectories::setRedIndex(int value)
{
	m_redSet = true;
	m_paramSpectrum.f1 = value;
	setCurrentImageIndex(m_currentImageIndex);
	emit frequencyChanged();
}

void FixedAttributImplFreeHorizonFromDirectories::setGreenIndex(int value)
{
	m_greenSet = true;
	m_paramSpectrum.f2 = value;
	setCurrentImageIndex(m_currentImageIndex);
	emit frequencyChanged();
}

void FixedAttributImplFreeHorizonFromDirectories::setBlueIndex(int value)
{
	m_blueSet = true;
	m_paramSpectrum.f3 = value;
	setCurrentImageIndex(m_currentImageIndex);
	emit frequencyChanged();
}

void FixedAttributImplFreeHorizonFromDirectories::setRGBIndexes(int r, int g, int b)
{
	m_redSet = true;
	m_greenSet = true;
	m_blueSet = true;
	m_paramSpectrum.f1 = r;
	m_paramSpectrum.f2 = g;
	m_paramSpectrum.f3 = b;
	setCurrentImageIndex(m_currentImageIndex);
	emit frequencyChanged();
}

void FixedAttributImplFreeHorizonFromDirectories::setGrayFreqIndexes(int idx)
{
	m_redSet = true;
	m_greenSet = true;
	m_blueSet = true;
	m_paramSpectrum.f1 = idx;
	m_paramSpectrum.f2 = idx;
	m_paramSpectrum.f3 = idx;
	setCurrentImageIndex(m_currentImageIndex);
}

int FixedAttributImplFreeHorizonFromDirectories::getNbreSpectrumFreq()
{
	QString attributName =  QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));
	if ( attributName == "spectrum" )
	{
		if ( m_nbreSpectrumFreq <= 0 )
		{
			QString filename = getAttributFileFromIndex(0);
			m_nbreSpectrumFreq = FreeHorizonManager::getNbreSpectrumFreq(filename.toStdString());
		}
		return m_nbreSpectrumFreq;
	}
	else return 0;
}


float FixedAttributImplFreeHorizonFromDirectories::getPasEch()
{
	if ( m_pasEch < 0.0f )
	{
		inri::Xt xt((char*)m_dataSetPath.toStdString().c_str());
		if ( !xt.is_valid() ) return 1.0;
		m_pasEch = xt.stepSamples();
	}
	return m_pasEch;
}

float FixedAttributImplFreeHorizonFromDirectories::getFrequency(int index)
{
	int Nfreq = getNbreSpectrumFreq();
	float pasEch = getPasEch();
	return 1000.0 * index / ( pasEch * (Nfreq-1) * 2.0 );

}

QString FixedAttributImplFreeHorizonFromDirectories::getLabelFromPosition(int index)
{
	float freq = getFrequency(index);
	QString label = QString::number(freq)+" Hz";
	return label;
}


/*
void FixedAttributImplFreeHorizonFromDirectories::trt_changeColor()
{
	QColorDialog dialog;
	dialog.setCurrentColor(getHorizonColor());
	dialog.setOption (QColorDialog::DontUseNativeDialog);
	if (dialog.exec() == QColorDialog::Accepted)
	{
		QColor color = dialog.currentColor();
		setHorizonColor(color);
		// item->setTextColor(0, color);
	}


}




void FixedAttributImplFreeHorizonFromDirectories::trt_properties()
{
	FileInformationWidget dialog(getIsoFileFromIndex(0));
	int code = dialog.exec();
}

void FixedAttributImplFreeHorizonFromDirectories::trt_location()
{
	QString cmd = "caja " + m_dirPath;
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}
*/
