


#include <QTimer>
#include <QFileInfo>
#include <QDir>
#include <QFileInfoList>
#include <QDebug>
#include <QCoreApplication>

#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>
#include "workingsetmanager.h"
#include "cudaimagepaletteholder.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include <rgtSpectrumHeader.h>

#include "fixedlayersfromdatasetandcubegraphicrepfactory.h"
#include <fixedlayerimplfreehorizonfromdatasetandcube.h>


FixedLayerImplFreeHorizonFromDatasetAndCube::FixedLayerImplFreeHorizonFromDatasetAndCube(
		QString dirPath, QString dirName, QString seismicName, WorkingSetManager *workingSet,
						const FixedLayersFromDatasetAndCube::Grid3DParameter &params, QObject *parent)
: FixedLayersFromDatasetAndCube("", dirName, workingSet, params, parent, false)
{
	m_dirPath = dirPath;
	m_dirName = dirName;
	m_seismicName = seismicName;

	m_width = params.width;
	m_depth= params.depth;
	m_heightFor3D = params.heightFor3D;
	m_sampleTransformation.reset(new AffineTransformation(*params.sampleTransformation));
	m_ijToXYTransfo.reset(new Affine2DTransformation(*params.ijToXYTransfo));
	m_ijToInlineXlineTransfoForInline.reset(new Affine2DTransformation(*params.ijToInlineXlineTransfoForInline));
	m_ijToInlineXlineTransfoForXline.reset(new Affine2DTransformation(*params.ijToInlineXlineTransfoForXline));
	m_uuid = QUuid::createUuid();
	m_cubeSeismicAddon = params.cubeSeismicAddon;
	m_name = dirName;
	m_isoStep = -25;
	m_modePlay = false;
	m_coef = 1;

	m_repFactory.reset(new FixedLayersFromDatasetAndCubeGraphicRepFactory(this));

	m_currentIso.reset(new CPUImagePaletteHolder(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));
	m_currentAttr.reset(new CPUImagePaletteHolder(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));
	loadIsoAndAttribute("");

	m_numLayers = 1;
	for (std::size_t i=0; i<m_numLayers; i++) {
		m_layers.push_back(QString::number(i*m_isoStep));
	}

	m_mode = READ;
	m_cacheFirstIso = 0;
	m_cacheLastIso = 0;
	m_cacheStepIso = 1;
	m_cacheFirstIndex = 0;
	m_cacheLastIndex = 0;
	m_cacheStepIndex = 1;

	m_defaultSimplifyMeshSteps = 10;
	m_simplifyMeshSteps = m_defaultSimplifyMeshSteps;
	m_compressionMesh=0;

	// QString surveyPath = workingSet->getManagerWidget()->get_survey_fullpath_name();

	setCurrentImageIndex(0);
	// m_timerRefresh = new QTimer();
	// connect(m_timerRefresh,SIGNAL(timeout()),this,SLOT(nextCurrentIndex()));
}



FixedLayerImplFreeHorizonFromDatasetAndCube::~FixedLayerImplFreeHorizonFromDatasetAndCube()
{

}


void FixedLayerImplFreeHorizonFromDatasetAndCube::loadIsoAndAttribute(QString attribute)
{
	QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_dirName.toStdString()));
	if ( type == "isochrone" )
	{
		m_attributName = m_dirPath + "/" + "isochrone.iso";
	}
	else
	{
		// compatibility with old format
		if ( m_dirPath.contains("ImportExport/IJK/HORIZONS") )
			m_attributName = m_dirPath + "/" + m_dirName + ".raw";
		else
			m_attributName = m_dirPath + "/" + m_dirName + QString::fromStdString(FreeHorizonManager::attributExt);// FreeHorizonQManager::attributSuffix;
	}

	m_isoName =  m_dirPath + "/" + "isochrone.iso";;
	m_numLayers = 1;
	m_isoOrigin = 0;
	m_isoStep = -1;
}

QString FixedLayerImplFreeHorizonFromDatasetAndCube::getAttributFileFromIndex(int index) {
	return m_attributName;
}

QString FixedLayerImplFreeHorizonFromDatasetAndCube::getIsoFileFromIndex(int index) {
	return m_isoName;
}


QString FixedLayerImplFreeHorizonFromDatasetAndCube::readAttributData(short *data, long size)
{
	std::string type = FreeHorizonManager::typeFromAttributName(m_dirName.toStdString());

	if ( type == "isochrone" )
	{
		FreeHorizonManager::readInt32(getAttributFileFromIndex(0).toStdString(), data);
	}
	else if ( type == "mean" )
	{
		FreeHorizonManager::readMean(getAttributFileFromIndex(0).toStdString(), data, size);
	}
	else if ( type == "gcc" )
	{
		FreeHorizonManager::readGcc(getAttributFileFromIndex(0).toStdString(), m_gccScale, data);
	}
	else
	{
		memset(data, 0, size*sizeof(short));
	}
	return "ok";
}

void FixedLayerImplFreeHorizonFromDatasetAndCube::getImageForIndex(long newIndex,
		CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return;

	QMutexLocker locker(&m_lock);
	//fprintf(stderr, "buffer ************************************** %d\n", newIndex, m_numLayers);

	// read rgb
	std::size_t w = width();
	std::size_t h = depth();
	std::size_t layerSize = w * h;

	if (m_mode==CACHE && ((newIndex-m_cacheFirstIndex)%m_cacheStepIndex)==0 && ((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)>0 &&
			((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)<((m_cacheLastIndex-m_cacheFirstIndex)/m_cacheStepIndex)) {
		long cacheRelativeIndex = (newIndex-m_cacheFirstIndex)/m_cacheStepIndex;

		std::list<SurfaceCache>::iterator it = m_cacheList.begin();
		std::advance(it, cacheRelativeIndex);
		attrCudaBuffer->updateTexture(it->attr.data(), false, it->attrRange, it->attrHistogram);
		isoCudaBuffer->updateTexture(it->iso.data(), false);
	} else {
		std::vector<short> buf;
		std::vector<short> outBuf;
		buf.resize(layerSize*2);
		outBuf.resize(layerSize);

		readAttributData(outBuf.data(), layerSize);
		attrCudaBuffer->updateTexture(outBuf.data(), false);

		FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), outBuf.data());
		// readAttributData(outBuf.data(), layerSize);
		// memset(outBuf.data(), 0, layerSize*sizeof(short));
		isoCudaBuffer->updateTexture(outBuf.data(), false);


		/*
		FILE *pf = fopen((char*)getAttributFileFromIndex(newIndex).toStdString().c_str(), "r");
		if ( pf )
		{
			if ( type == "gcc" )
			{

			}
			else
			{
				if ( pf != nullptr )
				{
					fread(outBuf.data(), sizeof(short), layerSize, pf);
					fclose(pf);
					attrCudaBuffer->updateTexture(outBuf.data(), false);
				}
			}
		}
		*/
	}

	if (m_isLockUsed) {
			attrCudaBuffer->setRange(m_lockedRange);
			attrCudaBuffer->setLookupTable(m_lockedLookupTable);
		}
}


bool FixedLayerImplFreeHorizonFromDatasetAndCube::getImageForIndex(long newIndex,
		QByteArray& attrBuffer, QByteArray& isoBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return false;

	fprintf(stderr, "buffer ************************************** %d\n", newIndex, m_numLayers);

	QMutexLocker locker(&m_lock);

	// read rgb
	std::size_t w = width();
	std::size_t h = depth();

	QByteArray buf;
	std::size_t layerSize = w * h;
	attrBuffer.resize(layerSize*sizeof(short));
	isoBuffer.resize(layerSize*sizeof(short));

	bool isValid = checkValidity<short>(attrBuffer, layerSize);
	isValid = isValid && checkValidity<short>(isoBuffer, layerSize);

	if (isValid && m_mode==CACHE && ((newIndex-m_cacheFirstIndex)%m_cacheStepIndex)==0 && ((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)>0 &&
			((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)<((m_cacheLastIndex-m_cacheFirstIndex)/m_cacheStepIndex)) {
		long cacheRelativeIndex = (newIndex-m_cacheFirstIndex)/m_cacheStepIndex;

		std::list<SurfaceCache>::iterator it = m_cacheList.begin();
		std::advance(it, cacheRelativeIndex);
		memcpy(attrBuffer.data(), it->attr.constData(), sizeof(short) * layerSize);
		memcpy(isoBuffer.data(), it->iso.constData(), sizeof(short) * layerSize);
	} else if (isValid) {
		buf.resize(layerSize*2*sizeof(short));
		isValid = checkValidity<short>(buf, layerSize*2);

		if (isValid) {
			short* attributTab = static_cast<short*>(static_cast<void*>(attrBuffer.data()));
			short* isoTab = static_cast<short*>(static_cast<void*>(isoBuffer.data()));

			FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)isoTab);
			// readAttributData(attributTab, layerSize);
			// memset(isoTab, 0, layerSize*sizeof(short));
			readAttributData(attributTab, layerSize);
		}
	}
	return isValid;
}


void FixedLayerImplFreeHorizonFromDatasetAndCube::setCurrentImageIndexInternal(long newIndex) {
	if (newIndex<0 || newIndex>=m_numLayers) {
		m_currentImageIndex = -1;
		return;
	}
	if (m_currentImageIndex==newIndex) {
		// return;
	} else {
		m_currentImageIndex = newIndex;
	}

	fprintf(stderr, "buffer2 ************************************** %d\n", newIndex, m_numLayers);

	if (m_currentImageIndex!=-1) {
		// read rgb
		QMutexLocker locker(&m_lock);
		std::size_t w = width();
		std::size_t h = depth();

		if (m_mode==CACHE && ((newIndex-m_cacheFirstIndex)%m_cacheStepIndex)==0 && ((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)>=0 &&
				((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)<=((m_cacheLastIndex-m_cacheFirstIndex)/m_cacheStepIndex)) {
			long cacheRelativeIndex = (newIndex-m_cacheFirstIndex)/m_cacheStepIndex;

			std::list<SurfaceCache>::iterator it = m_cacheList.begin();
			std::advance(it, cacheRelativeIndex);
			m_currentAttr->updateTexture(it->attr, false, it->attrRange, it->attrHistogram);
			m_currentIso->updateTexture(it->iso, false);
		} else {

			// std::vector<short> buf;
			QByteArray outBuf;
			std::size_t layerSize = w * h;
			// buf.resize(layerSize*2);
			outBuf.resize(layerSize*sizeof(short));
			short* outTab = static_cast<short*>(static_cast<void*>(outBuf.data()));

			// readAttributData(outTab, layerSize);

			// FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)outTab);
			readAttributData(outTab, layerSize);
			m_currentAttr->updateTexture(outBuf, false);

			/*
			outTab = static_cast<short*>(static_cast<void*>(outBuf.data()));
			FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)outTab);
			m_currentIso->updateTexture(outBuf, false);
			*/

			QByteArray outBufF;
			outBufF.resize(layerSize*sizeof(float));
			float *outTabF = static_cast<float*>(static_cast<void*>(outBufF.data()));
			FreeHorizonManager::read(getIsoFileFromIndex(newIndex).toStdString(), (float*)outTabF);
			m_currentIso->updateTexture(outBufF, false);
		}

		// apply locked palette should not trigger recompute (range+histo) if data range does not change
		if (m_isLockUsed) {
			m_currentAttr->setRange(m_lockedRange);
			m_currentAttr->setLookupTable(m_lockedLookupTable);
		}

		emit currentIndexChanged(m_currentImageIndex);
	}
}


void FixedLayerImplFreeHorizonFromDatasetAndCube::setCurrentImageIndex(long newIndex) {
	if (!isInitialized()) {
		m_currentImageIndex = newIndex;
		return;
	}

	if (m_lockRead.tryLock()) {
		{
			QMutexLocker locker(&m_lockNextRead);
			m_nextIndex = -1;
		}
		bool goOn = true;
		long nextIndex = newIndex;
		while(goOn) {
			setCurrentImageIndexInternal(nextIndex);
			QCoreApplication::processEvents();
			QMutexLocker locker(&m_lockNextRead);
			goOn = m_nextIndex>=0;
			if (goOn) {
				nextIndex = m_nextIndex;
				m_nextIndex = -1;
			}
		}
		m_lockRead.unlock();
	} else {
		QMutexLocker locker(&m_lockNextRead);
		m_nextIndex = newIndex;
	}
}


QColor FixedLayerImplFreeHorizonFromDatasetAndCube::getHorizonColor()
{
	bool ok = false;
	QColor col = FreeHorizonQManager::loadColorFromPath(m_dirPath, &ok);
	if ( ok ) return col;
	return QColor(Qt::white);
}

void FixedLayerImplFreeHorizonFromDatasetAndCube::setHorizonColor(QColor col)
{
	FreeHorizonQManager::saveColorToPath(m_dirPath, col);
	emit colorChanged(col);
}


bool FixedLayerImplFreeHorizonFromDatasetAndCube::enableScaleSlider()
{
	std::string type = FreeHorizonManager::typeFromAttributName(m_dirName.toStdString());
	if ( type.compare("gcc") == 0 ) return true;
	return false;
}

int FixedLayerImplFreeHorizonFromDatasetAndCube::getNbreGccScales()
{
	if ( m_gccNbreScales < 0 )
	{
		m_gccNbreScales = FreeHorizonManager::getNbreGccScales(getAttributFileFromIndex(0).toStdString());
	}
	return m_gccNbreScales;
}


void FixedLayerImplFreeHorizonFromDatasetAndCube::setGccIndex(int value)
{
	m_gccScale = value;
	setCurrentImageIndex(m_currentImageIndex);
}
