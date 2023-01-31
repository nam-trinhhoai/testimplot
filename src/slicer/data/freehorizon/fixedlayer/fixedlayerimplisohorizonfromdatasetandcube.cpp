
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
#include <fixedlayerimplisohorizonfromdatasetandcube.h>


FixedLayerImplIsoHorizonFromDatasetAndCube::FixedLayerImplIsoHorizonFromDatasetAndCube(
		QString dirPath, QString dirName, QString seismicName, WorkingSetManager *workingSet,
						const FixedLayersFromDatasetAndCube::Grid3DParameter &params, QObject *parent)
: FixedLayersFromDatasetAndCube("", dirName, workingSet, params, parent)
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
	setCurrentImageIndex(0);

	m_timerRefresh = new QTimer();
	connect(m_timerRefresh,SIGNAL(timeout()),this,SLOT(nextCurrentIndex()));
}



FixedLayerImplIsoHorizonFromDatasetAndCube::~FixedLayerImplIsoHorizonFromDatasetAndCube()
{

}


void FixedLayerImplIsoHorizonFromDatasetAndCube::loadIsoAndAttribute(QString attribute)
{
	QDir mainDir(m_dirPath);
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
			if ( m_dirName == ISODATA_NAME || m_dirName == ISODATA_OLDNAME )
			{
				attributName = m_dirName + ".iso";
			}
			else
			{
				attributName = m_dirName + QString::fromStdString(FreeHorizonManager::attributExt);// FreeHorizonQManager::attributSuffix;
			}
			m_attributNames[i] = mainDirList[i].absoluteFilePath() + "/" + attributName;
			m_isoNames[i] = mainDirList[i].absoluteFilePath() + "/" + ISODATA_NAME + ".iso";
		}
		m_numLayers = N;
		m_isoOrigin = N;
		m_isoStep = -1;
	}
}

QString FixedLayerImplIsoHorizonFromDatasetAndCube::getAttributFileFromIndex(int index) {
	return m_attributNames[index];
}

QString FixedLayerImplIsoHorizonFromDatasetAndCube::getIsoFileFromIndex(int index) {
	return m_isoNames[index];
}



void FixedLayerImplIsoHorizonFromDatasetAndCube::getImageForIndex(long newIndex,
		CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return;

	QMutexLocker locker(&m_lock);
//	fprintf(stderr, "buffer ************************************** %d\n", newIndex, m_numLayers);

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

		std::string type = FreeHorizonManager::typeFromAttributName(m_dirName.toStdString());

		if ( type == "isochrone" )
		{
			FreeHorizonManager::readInt32(getAttributFileFromIndex(newIndex).toStdString(), outBuf.data());
		}
		else if ( type == "mean" )
		{
			FreeHorizonManager::readMean(getAttributFileFromIndex(newIndex).toStdString(), outBuf.data(), layerSize);
		}
		attrCudaBuffer->updateTexture(outBuf.data(), false);


		std::vector<float> bufF;
		std::vector<float> outBufF;
		bufF.resize(layerSize*2);
		outBufF.resize(layerSize);
		FreeHorizonManager::read(getIsoFileFromIndex(newIndex).toStdString(), outBufF.data());
		isoCudaBuffer->updateTexture(outBufF.data(), false);
	}

	if (m_isLockUsed) {
			attrCudaBuffer->setRange(m_lockedRange);
			attrCudaBuffer->setLookupTable(m_lockedLookupTable);
		}
}


bool FixedLayerImplIsoHorizonFromDatasetAndCube::getImageForIndex(long newIndex,
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

			FreeHorizonManager::readInt32(getIsoFileFromIndex(newIndex).toStdString(), (short*)attributTab);
			std::string type = FreeHorizonManager::typeFromAttributName(m_dirName.toStdString());

			if ( type == "isochrone" )
			{
				FreeHorizonManager::readInt32(getAttributFileFromIndex(newIndex).toStdString(), isoTab);
			}
			else if ( type == "mean" )
			{
				FreeHorizonManager::readMean(getAttributFileFromIndex(newIndex).toStdString(), isoTab, layerSize);
			}
		}
	}
	return isValid;
}


void FixedLayerImplIsoHorizonFromDatasetAndCube::setCurrentImageIndexInternal(long newIndex) {
	if (newIndex<0 || newIndex>=m_numLayers) {
		m_currentImageIndex = -1;
		return;
	}
	if (m_currentImageIndex==newIndex) {
		return;
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

			std::vector<short> buf;
			QByteArray outBuf;
			std::size_t layerSize = w * h;
			buf.resize(layerSize*2);
			outBuf.resize(layerSize*sizeof(short));
			short* outTab = static_cast<short*>(static_cast<void*>(outBuf.data()));
			std::string type = FreeHorizonManager::typeFromAttributName(m_dirName.toStdString());

			if ( type == "isochrone" )
			{
				FreeHorizonManager::readInt32(getAttributFileFromIndex(newIndex).toStdString(), outTab);
			}
			else if ( type == "mean" )
			{
				FreeHorizonManager::readMean(getAttributFileFromIndex(newIndex).toStdString(), outTab, layerSize);
			}
			m_currentAttr->updateTexture(outBuf, false);

			std::vector<float> bufF;
			QByteArray outBufF;
			bufF.resize(layerSize*2);
			outBufF.resize(layerSize*sizeof(float));
			float* outTabF = static_cast<float*>(static_cast<void*>(outBufF.data()));
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


void FixedLayerImplIsoHorizonFromDatasetAndCube::setCurrentImageIndex(long newIndex) {
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


QColor FixedLayerImplIsoHorizonFromDatasetAndCube::getHorizonColor()
{
	bool ok = false;
	QColor col = FreeHorizonQManager::loadColorFromPath(m_dirPath, &ok);
	if ( ok ) return col;
	return QColor(Qt::white);
}

void FixedLayerImplIsoHorizonFromDatasetAndCube::setHorizonColor(QColor col)
{
	FreeHorizonQManager::saveColorToPath(m_dirPath, col);
	emit colorChanged(col);
}
