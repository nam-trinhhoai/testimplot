#include "fixedlayersfromdatasetandcube.h"

#include "fixedlayersfromdatasetandcubegraphicrepfactory.h"
#include "igraphicrepfactory.h"
#include "nvhorizontransformgenerator.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "ijkhorizon.h"
#include "affinetransformation.h"
#include "gdalloader.h"
#include "workingsetmanager.h"
#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"
#include "folderdata.h"
#include "seismicsurvey.h"
#include "smdataset3D.h"
#include "sismagedbmanager.h"
#include "SeismicManager.h"
#include "gdal.h"
#include "datasetrelatedstorageimpl.h"
#include "GeotimeProjectManagerWidget.h"
#include "stringselectordialog.h"
#include "surfacemesh.h"
#include "cpuimagepaletteholder.h"
#include "affine2dtransformation.h"

#include <array>

#include <gdal_priv.h>
#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QFileDialog>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QMessageBox>


// cudaBuffer need to be a float RGBD planar stack
FixedLayersFromDatasetAndCube::FixedLayersFromDatasetAndCube(QString cube,
			QString name, WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent,
			bool valide) : IData(workingSet, parent) {
	if ( !valide ) return;
	m_width = params.width;
	m_depth= params.depth;
	m_heightFor3D = params.heightFor3D;
	m_sampleTransformation.reset(new AffineTransformation(*params.sampleTransformation));
	m_ijToXYTransfo.reset(new Affine2DTransformation(*params.ijToXYTransfo));
	m_ijToInlineXlineTransfoForInline.reset(new Affine2DTransformation(*params.ijToInlineXlineTransfoForInline));
	m_ijToInlineXlineTransfoForXline.reset(new Affine2DTransformation(*params.ijToInlineXlineTransfoForXline));
	m_uuid = QUuid::createUuid();
	m_cubeSeismicAddon = params.cubeSeismicAddon;
	m_name = name;
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

	loadIsoAndAttribute(cube);

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

FixedLayersFromDatasetAndCube::~FixedLayersFromDatasetAndCube() {
	if (m_f!=nullptr) {
		fclose(m_f);
	}
}

void FixedLayersFromDatasetAndCube::loadIsoAndAttribute(QString attribute) {
	qDebug() << attribute;
	QFileInfo fileInfo(attribute);

	if ( fileInfo.isFile() )
	{
		m_f = fopen(attribute.toStdString().c_str(), "r");
		m_attrPath = attribute;
		std::size_t sz;
		bool isValid;
		if (m_f!=nullptr) {
			fseek(m_f, 0L, SEEK_END);
			sz = ftell(m_f);
			fseek(m_f, 0L, SEEK_SET);
			isValid = true;
		}

		if (isValid) {
			m_numLayers = sz / (width() * depth() * sizeof(short) * 2 );
		} else {
			m_numLayers = 0;
		}
		m_isoOrigin = (m_numLayers-1) * (-m_isoStep);
		m_dataType = 0;
	}
	else if ( fileInfo.isDir())
	{
		QDir dir(attribute);
		QFileInfoList dirList = dir.entryInfoList(QStringList() << "iso_*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
		if ( dirList.size() > 0 )
		{
			m_dirNames.clear();
			m_dirNames.resize(dirList.size());
			for (int i=0; i<dirList.size(); i++)
				m_dirNames[i] = dirList[i].absoluteFilePath();
			m_numLayers = m_dirNames.size();
			// m_isoOrigin = 32000;
			m_isoOrigin = (m_numLayers-1) * (-m_isoStep);
			m_dataType = 1;
		}
		else
		{
			qDebug() << "FixedRGBLayersFromDatasetAndCube : Invalid dirPath : " << attribute;
			m_numLayers = 0;
			m_isoOrigin = 0;
		}
	}
	else
	{
		m_numLayers = 0;
		qDebug() << "unknow format for: " << attribute;
	}
}

QString FixedLayersFromDatasetAndCube::getCurrentObjFile() const
{
	return getObjFile(m_currentImageIndex);
}

QString FixedLayersFromDatasetAndCube::getObjFile(int index) const
{
	QFileInfo fileinfo(m_attrPath);
	QString temp = fileinfo.completeBaseName();
	qDebug()<<"fileinfo "<<temp;

	QDir dir = fileinfo.absoluteDir();
	if(!dir.exists(temp))
	{
		dir.mkdir(temp);
	}

	dir.cd(temp);

	QString res = QString::number(index)+".obj";//3do

	return dir.absoluteFilePath(res);
}

SurfaceMeshCache* FixedLayersFromDatasetAndCube::getMeshCache(int newIndex)
{
	if (isIndexCache(newIndex))
	{
		long cacheRelativeIndex = (newIndex-m_cacheFirstIndex)/m_cacheStepIndex;

		std::list<SurfaceCache>::iterator it = m_cacheList.begin();
		std::advance(it, cacheRelativeIndex);

		return &it->meshCache;



	}
	return nullptr;
}

bool FixedLayersFromDatasetAndCube::isIndexCache(int newIndex)
{
	return (m_mode==CACHE && ((newIndex-m_cacheFirstIndex)%m_cacheStepIndex)==0 && ((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)>=0 &&
							((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)<=((m_cacheLastIndex-m_cacheFirstIndex)/m_cacheStepIndex));
	//return (index >0 && index < m_numLayers);
}

int FixedLayersFromDatasetAndCube::getSimplifyMeshSteps()const
{
	return m_simplifyMeshSteps;
}
void FixedLayersFromDatasetAndCube::setSimplifyMeshSteps(int steps)
{
	if(m_simplifyMeshSteps  != steps)
	{
		m_simplifyMeshSteps  = steps;
		emit simplifyMeshStepsChanged(steps);
	}
}



int FixedLayersFromDatasetAndCube::getCompressionMesh()const
{
	return m_compressionMesh;
}

void FixedLayersFromDatasetAndCube::setCompressionMesh(int compress)
{
	if(m_compressionMesh  != compress)
	{
		m_compressionMesh  = compress;
		emit compressionMeshChanged(compress);
	}
}

unsigned int FixedLayersFromDatasetAndCube::width() const {
	return m_width;
}

unsigned int FixedLayersFromDatasetAndCube::depth() const {
	return m_depth;
}

unsigned int FixedLayersFromDatasetAndCube::getNbProfiles() const {
	return depth();
}

unsigned int FixedLayersFromDatasetAndCube::getNbTraces() const {
	return width();
}

unsigned int FixedLayersFromDatasetAndCube::heightFor3D() const { // only use for 3d
	return m_heightFor3D;
}

float FixedLayersFromDatasetAndCube::getStepSample() {
	return m_sampleTransformation->a();
}

float FixedLayersFromDatasetAndCube::getOriginSample() {
	return m_sampleTransformation->b();
}

//IData
IGraphicRepFactory* FixedLayersFromDatasetAndCube::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid FixedLayersFromDatasetAndCube::dataID() const {
	return m_uuid;
}

QString FixedLayersFromDatasetAndCube::name() const {
	return m_name;
}

std::size_t FixedLayersFromDatasetAndCube::numLayers() {
	return m_numLayers;
}


const QList<QString>& FixedLayersFromDatasetAndCube::layers() const {
	return m_layers;
}

long FixedLayersFromDatasetAndCube::currentImageIndex() const {
	return m_currentImageIndex;
}

template<typename ValType>
void FixedLayersFromDatasetAndCube::swap(ValType& val) {
	char* tab = (char*) &val;
	char tmp;
	long N = sizeof(ValType);
	for (long i=0; i<N/2; i++) {
		tmp = tab[i];
		tab[i] = tab[N-1-i];
		tab[N-1-i] = tmp;
	}
}

QString FixedLayersFromDatasetAndCube::getIsoFileFromIndex(int index) const {
	QString res = m_dirNames[index] + "/isoData.raw";
	qDebug() << res;
	return res;
}

QString FixedLayersFromDatasetAndCube::getMeanFileFromIndex(int index) const {
	QString res = m_dirNames[index] + "/mean.raw";
	return res;
}


void FixedLayersFromDatasetAndCube::getImageForIndex(long newIndex,
		CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return;

	QMutexLocker locker(&m_lock);

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

		if ( m_dataType == 0 )
		{
			size_t absolutePosition = layerSize * newIndex * sizeof(short) * 2;
			{
				QMutexLocker b2(&m_lockFile);
				fseek(m_f, absolutePosition, SEEK_SET);
				fread(buf.data(), sizeof(short), layerSize*2, m_f);
			}
			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*2];
				// swap
				//swap(val);
				outBuf[idx] = val;
			}
			attrCudaBuffer->updateTexture(outBuf.data(), false);

			for (std::size_t idx=0; idx<layerSize; idx++) {
				short val = buf[idx*2+1];
				//swap(val);
				outBuf[idx] = val;
			}
			isoCudaBuffer->updateTexture(outBuf.data(), false);
		}
		else
		{
			FILE *pf = fopen((char*)getMeanFileFromIndex(newIndex).toStdString().c_str(), "r");
			if ( pf != nullptr )
			{
				fread(outBuf.data(), sizeof(short), layerSize, pf);
				attrCudaBuffer->updateTexture(outBuf.data(), false);
				fclose(pf);
			}
			pf = fopen((char*)getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
			if ( pf != nullptr )
			{
				float *tmp = (float*)calloc(layerSize, sizeof(float));
				fread(tmp, sizeof(float), layerSize, pf);

				for (long idx=0; idx<layerSize; idx++) outBuf[idx] = (short)tmp[idx];
				free(tmp);
				fclose(pf);
				isoCudaBuffer->updateTexture(outBuf.data(), false);
			}
		}
	}

	// apply locked palette should not trigger recompute (range+histo) if data range does not change
	if (m_isLockUsed) {
		attrCudaBuffer->setRange(m_lockedRange);
		attrCudaBuffer->setLookupTable(m_lockedLookupTable);
	}
}

bool FixedLayersFromDatasetAndCube::getImageForIndex(long newIndex,
		QByteArray& attrBuffer, QByteArray& isoBuffer) {
	if (newIndex<0 || newIndex>=m_numLayers)
		return false;

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

			if ( m_dataType == 0 )
			{
				size_t absolutePosition = layerSize * newIndex * sizeof(short) * 2;
				{
					QMutexLocker b2(&m_lockFile);
					fseek(m_f, absolutePosition, SEEK_SET);
					fread(buf.data(), sizeof(short), layerSize*2, m_f);
				}
				short* attributTab = static_cast<short*>(static_cast<void*>(attrBuffer.data()));
				const short* bufTab = static_cast<const short*>(static_cast<const void*>(buf.constData()));
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = bufTab[idx*2];
					// swap
					//swap(val);

					attributTab[idx] = val;
				}
				short* isoTab = static_cast<short*>(static_cast<void*>(isoBuffer.data()));
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = bufTab[idx*2+1];
					//swap(val);
					isoTab[idx] = val;
				}
			}
			else
			{
				short* attributTab = static_cast<short*>(static_cast<void*>(attrBuffer.data()));
				short* isoTab = static_cast<short*>(static_cast<void*>(isoBuffer.data()));
				FILE *pf = fopen((char*)getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if ( pf != nullptr )
				{
					float *tmp = (float*)calloc(layerSize, sizeof(float));
					fread(tmp, sizeof(float), layerSize, pf);
					for (long idx=0; idx<layerSize; idx++) attributTab[idx] = (short)tmp[idx];
					free(tmp);
					fclose(pf);
				}

				pf = fopen((char*)getMeanFileFromIndex(newIndex).toStdString().c_str(), "r");
				if ( pf != nullptr )
				{
					fread(isoTab, sizeof(short), layerSize, pf);
					fclose(pf);
				}
			}
		}
	}
	return isValid;
}

void FixedLayersFromDatasetAndCube::nextCurrentIndex()
{
	if(m_modePlay)
		{
			if (m_mode==CACHE)
			{
				int indexmin = qMin (m_cacheFirstIndex,m_cacheLastIndex);
				int indexmax = qMax (m_cacheFirstIndex,m_cacheLastIndex);
				int step = qAbs(m_cacheStepIndex);
				if(m_loop)
				{
					if( currentImageIndex() >= indexmax)
					{
						m_incr = -1;
					}
					else if( currentImageIndex() <= indexmin)
					{
						m_incr = 1;
					}
				}
				else
				{
					m_incr = 1;
				}

				int index = (currentImageIndex()+(step*m_incr) -indexmin) %( indexmax - indexmin +1);
				setCurrentImageIndex(index+indexmin);
			}
			else
			{
				if(m_loop)
				{
					if( m_incr > 0)
					{
						if(currentImageIndex() == numLayers()-1)
						{
							m_incr = -1;
						}
					}
					else if( m_incr < 0)
					{
						if(currentImageIndex() ==0)
							m_incr = 1;
					}
				}
				else
				{
					m_incr = 1;
				}
				int index = (currentImageIndex() + m_incr*m_coef) %numLayers();
				setCurrentImageIndex(index);
			}
		}
}


void FixedLayersFromDatasetAndCube::play(int interval,int coef,bool looping)
{
	m_modePlay = !m_modePlay;
	m_loop = looping;
	m_coef = coef;
	if(m_modePlay)
	{
		if(numLayers() > 0)
			m_timerRefresh->start(interval);
		else
			qDebug()<<"Error numLayers = 0! ";
	}
	else
	{
		m_timerRefresh->stop();
	}

}

void FixedLayersFromDatasetAndCube::setCurrentImageIndex(long newIndex) {
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

void FixedLayersFromDatasetAndCube::setCurrentImageIndexInternal(long newIndex) {
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

			if ( m_dataType == 0 )
			{
				size_t absolutePosition = layerSize * m_currentImageIndex * sizeof(short) * 2;
				{
					QMutexLocker b2(&m_lockFile);
					fseek(m_f, absolutePosition, SEEK_SET);
					fread(buf.data(), sizeof(short), layerSize*2, m_f);
				}
				short* outTab = static_cast<short*>(static_cast<void*>(outBuf.data()));
				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = buf[idx*2];
					outTab[idx] = val;
				}
				m_currentAttr->updateTexture(outBuf, false);

				for (std::size_t idx=0; idx<layerSize; idx++) {
					short val = buf[idx*2+1];
					outTab[idx] = val;
				}
				m_currentIso->updateTexture(outBuf, false);
			}
			else
			{
				short* outTab = static_cast<short*>(static_cast<void*>(outBuf.data()));
				FILE *pf = fopen((char*)getMeanFileFromIndex(newIndex).toStdString().c_str(), "r");
				if ( pf != nullptr )
				{
					fread(outTab, sizeof(short), layerSize, pf);
					fclose(pf);
					m_currentAttr->updateTexture(outBuf, false);
				}

				pf = fopen((char*)getIsoFileFromIndex(newIndex).toStdString().c_str(), "r");
				if ( pf != nullptr )
				{
					float *tmp = (float*)calloc(layerSize, sizeof(float));
					fread(tmp, sizeof(float), layerSize, pf);
					for (long idx=0; idx<layerSize; idx++) outTab[idx] = (short)tmp[idx];
					free(tmp);
					fclose(pf);
					m_currentIso->updateTexture(outBuf, false);
				}
			}
		}

		// apply locked palette should not trigger recompute (range+histo) if data range does not change
		if (m_isLockUsed) {
			m_currentAttr->setRange(m_lockedRange);
			m_currentAttr->setLookupTable(m_lockedLookupTable);
		}

		emit currentIndexChanged(m_currentImageIndex);
	}
}

FixedLayersFromDatasetAndCube* FixedLayersFromDatasetAndCube::createDataFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, QObject *parent) {
	QString cubeFile;
	QString cubeName;

	QStringList meanPaths;
	QStringList meanNames;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";

	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				QFileInfoList meanInfoList = dir.entryInfoList(QStringList() << "*mean*raw", QDir::Files | QDir::Readable);
				for (const QFileInfo& meanInfo : meanInfoList) {
					meanPaths << meanInfo.absoluteFilePath();
					meanNames << meanInfo.fileName();
				}
				// ===================== new dir
				qDebug() << fileInfo.absoluteFilePath();
				QString mainDir = fileInfo.absoluteFilePath() + "/cubeRgt2RGB/";
				QDir dir(mainDir);
				QFileInfoList dirList0 = dir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
				for (int i=0; i<dirList0.size(); i++)
				{
					QString mainDir2 = dirList0[i].absoluteFilePath();
					QDir dir2(mainDir2);
					QFileInfoList dirList2 = dir2.entryInfoList(QStringList() << "iso_*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
					if ( dirList2.size() > 0 )
					{
						meanPaths << mainDir2;
						meanNames << dirList0[i].fileName();
					}
					qDebug() << dirList0[i].absoluteFilePath();
				}
			}
		}
	}
	bool isValid = meanNames.count()>0;
	bool errorLogging = false;
	int meanIndex = 0;
	if (isValid) {
		QStringList meanNamesBuf = meanNames;
		StringSelectorDialog dialog(&meanNamesBuf, "Select mean");
		int result = dialog.exec();
		meanIndex = dialog.getSelectedIndex();

		isValid = result==QDialog::Accepted && meanIndex<meanNames.count() && meanIndex>=0;
	} else {
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any Mean data");
		errorLogging = true;
	}

	FixedLayersFromDatasetAndCube* outObj = nullptr;
	QString sismageName;
	if (isValid) {
		QDir dir = QFileInfo(meanPaths[meanIndex]).dir();
		isValid = dir.cdUp();
		sismageName = dir.dirName();
	}
	if (isValid) {
		cubeFile = meanPaths[meanIndex];
		cubeName = meanNames[meanIndex];
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params;

			// get transforms
			SmDataset3D d3d(datasetPath.toStdString());
			AffineTransformation sampleTransfo = d3d.sampleTransfo();
			Affine2DTransformation inlineXlineTransfoForInline = d3d.inlineXlineTransfoForInline();
			Affine2DTransformation inlineXlineTransfoForXline = d3d.inlineXlineTransfoForXline();

			// params.sampleTransformation = &sampleTransfo;
			// params.ijToInlineXlineTransfoForInline = &inlineXlineTransfoForInline;
			// params.ijToInlineXlineTransfoForXline = &inlineXlineTransfoForXline;
			params.sampleTransformation = std::make_shared<AffineTransformation>(sampleTransfo);
			params.ijToInlineXlineTransfoForInline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForInline);
			params.ijToInlineXlineTransfoForXline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForXline);

			std::array<double, 6> inlineXlineTransfo =
					survey->inlineXlineToXYTransfo()->direct();
			std::array<double, 6> ijToInlineXline = d3d.inlineXlineTransfo().direct();

			std::array<double, 6> res;
			GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
					res.data());

			Affine2DTransformation ijToXYTransfo(d3d.inlineXlineTransfo().width(),
					d3d.inlineXlineTransfo().height(), res);
			// params.ijToXYTransfo = &ijToXYTransfo;
			params.ijToXYTransfo = std::make_shared<Affine2DTransformation>(ijToXYTransfo);


			inri::Xt xt(datasetPath.toStdString().c_str());
			if (xt.is_valid()) {
				params.cubeSeismicAddon.set(
							xt.startSamples(), xt.stepSamples(),
							xt.startRecord(), xt.stepRecords(),
							xt.startSlice(), xt.stepSlices());
				params.width = xt.nRecords();
				params.depth = xt.nSlices();
				params.heightFor3D = xt.nSamples();
				int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(datasetPath);
				params.cubeSeismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);
			} else {
				std::cerr << "xt cube is not valid (" << datasetPath.toStdString() << ")" << std::endl;
				isValid = false;
			}

			if (isValid) {
				QStringList splited = cubeName.split("_");
				QString dataName = sismageName;
				int N = splited.count();
				if (N-2>0 && splited[N-2].compare("size")==0) {
					dataName = extractListAndJoin(splited, 0, N-3, "_");
				}
				outObj = new FixedLayersFromDatasetAndCube(
							cubeFile, prefix+dataName+" (mean)", workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

QString FixedLayersFromDatasetAndCube::extractListAndJoin(QStringList list, long beg, long end, QString joinStr) {
	QStringList newList;
	for (long i=beg; i<=end; i++) {
		newList << list[i];
	}
	return newList.join(joinStr);
}

FixedLayersFromDatasetAndCube::Mode FixedLayersFromDatasetAndCube::mode() const {
	return m_mode;
}

void FixedLayersFromDatasetAndCube::moveToReadMode() {
	if (m_mode != READ) {
		QMutexLocker locker(&m_lock);
		m_mode = READ;
		m_cacheFirstIso = 0;
		m_cacheLastIso = 0;
		m_cacheStepIso = 1;
		m_cacheList.clear();
	}
	emit modeChanged();
}

bool FixedLayersFromDatasetAndCube::moveToCacheMode(long firstIso, long lastIso, long isoStep) {
	bool out = false;
	if (m_mode != CACHE) {
		// fix lastIso
		lastIso = ((lastIso - firstIso) / isoStep) * isoStep + firstIso;

		bool areArgumentsValid = ((m_isoOrigin-firstIso) % m_isoStep)==0 && (isoStep%m_isoStep)==0 && isoStep!=0;
		long firstIndex = (firstIso - m_isoOrigin) / m_isoStep;
		areArgumentsValid = areArgumentsValid && firstIndex>=0 && firstIndex<m_numLayers;
		long lastIndex = (lastIso - m_isoOrigin) / m_isoStep;
		areArgumentsValid = areArgumentsValid && lastIndex>=0 && lastIndex<m_numLayers;
		if (areArgumentsValid) {
			// reorganize first last and step
			long indexInc = isoStep / m_isoStep;
			if ((firstIndex<lastIndex && indexInc<0) || (firstIndex>lastIndex && indexInc>0)) {
				indexInc = -indexInc;
				isoStep = -isoStep;
			}

			int mapSize = width() * depth();

			emit initProgressBar(firstIndex * indexInc, lastIndex * indexInc, firstIndex * indexInc);
			QCoreApplication::processEvents();
			bool loading = true;
			CUDAImagePaletteHolder paletteHolder(width(), depth(), ImageFormats::QSampleType::INT16);
			for (long index=firstIndex; (lastIndex-index)*indexInc>=0; index+=indexInc) {
				SurfaceCache cache;
				loading = getImageForIndex(index, cache.attr, cache.iso);
				if (!loading) {
					break;
				}

				paletteHolder.updateTexture(cache.attr.data(), false);
				cache.attrRange = paletteHolder.dataRange();
				cache.attrHistogram = paletteHolder.computeHistogram(cache.attrRange,
								QHistogram::HISTOGRAM_SIZE);

				float origin =0.0f;
				float scal = 1.0f;

				if(!isIsoInT())
				{
					origin =m_sampleTransformation->b();
					scal = m_sampleTransformation->a();
				}

				const float* tbuf = m_ijToXYTransfo->imageToWorldTransformation().constData();
					QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
													 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
													 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
													 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);
				//SurfaceMeshCache& cache,const std::vector<short>& isobuffer ,int width, int depth ,float steps, float cubeOrigin,float cubeScale,QString path, QMatrix4x4 transform
				const short* isotab = static_cast<const short*>(static_cast<const void*>(cache.iso.constData()));
				SurfaceMesh::createCache<short>(cache.meshCache,isotab,m_width,m_depth,m_simplifyMeshSteps,origin,scal,getObjFile(index),ijToXYTranformSwapped,m_compressionMesh);
				m_cacheList.push_back(cache);
				emit valueProgressBarChanged(index * indexInc);
				QCoreApplication::processEvents();
			}
			emit endProgressBar();

			if (loading) {
				QMutexLocker locker(&m_lock);
				m_cacheFirstIso = firstIso;
				m_cacheLastIso = lastIso;
				m_cacheStepIso = isoStep;
				m_cacheFirstIndex = firstIndex;
				m_cacheLastIndex = lastIndex;
				m_cacheStepIndex = indexInc;
				m_mode = CACHE;

				out = true;
			} else {
				m_cacheList.clear();
				QMessageBox::warning(nullptr, "Cache Mode", "Fail to load into memory the defined cache. Not Enough Memory !");
				out = false;
			}
		}
	}
	emit modeChanged();
	return out;
}

long FixedLayersFromDatasetAndCube::cacheFirstIndex() const {
	return m_cacheFirstIndex;
}

long FixedLayersFromDatasetAndCube::cacheLastIndex() const {
	return m_cacheLastIndex;
}

long FixedLayersFromDatasetAndCube::cacheStepIndex() const {
	return m_cacheStepIndex;
}

QString FixedLayersFromDatasetAndCube::attributePath() const {
	return m_attrPath;
}

CubeSeismicAddon FixedLayersFromDatasetAndCube::cubeSeismicAddon() const {
	return m_cubeSeismicAddon;
}

const AffineTransformation* FixedLayersFromDatasetAndCube::sampleTransformation() const {
	return m_sampleTransformation.get();
}

const Affine2DTransformation* FixedLayersFromDatasetAndCube::ijToXYTransfo() const {
	return m_ijToXYTransfo.get();
}

const Affine2DTransformation* FixedLayersFromDatasetAndCube::ijToInlineXlineTransfoForInline() const {
	return m_ijToInlineXlineTransfoForInline.get();
}

const Affine2DTransformation* FixedLayersFromDatasetAndCube::ijToInlineXlineTransfoForXline() const {
	return m_ijToInlineXlineTransfoForXline.get();
}

void FixedLayersFromDatasetAndCube::lockPalette(const QVector2D& range, const LookupTable& lookupTable) {
	m_lockedRange = range;
	m_lockedLookupTable = lookupTable;
	m_isLockUsed = true;
}

void FixedLayersFromDatasetAndCube::unlockPalette() {
	m_isLockUsed = false;
}

bool FixedLayersFromDatasetAndCube::isPaletteLocked() const {
	return m_isLockUsed;
}

const QVector2D& FixedLayersFromDatasetAndCube::lockedRange() const {
	return m_lockedRange;
}

const LookupTable& FixedLayersFromDatasetAndCube::lockedLookupTable() const {
	return m_lockedLookupTable;
}

long long FixedLayersFromDatasetAndCube::cacheLayerMemoryCost() const {
	long long longWidth = m_width;
	long long longDepth = m_depth;
	// iso + attr as short
	long long cost = 4 * longWidth * longDepth;

	long long reducedWidth = (longWidth - 1) / m_defaultSimplifyMeshSteps + 1;
	long long reducedHeight = (longWidth - 1) / m_defaultSimplifyMeshSteps + 1;

	// vertex xyz + normal xyz as float + texture uv as unsigned int
	cost += 4 * 8 * reducedWidth * reducedHeight;

	// triangles p1 p2 and p3 as unsigned int
	cost += 4 * 6 * (reducedWidth - 1) * (reducedHeight - 1);
	return cost;
}

std::vector<StackType> FixedLayersFromDatasetAndCube::stackTypes() const {
	std::vector<StackType> typeVect;
	typeVect.push_back(StackType::ISO);
	return typeVect;
}

std::shared_ptr<AbstractStack> FixedLayersFromDatasetAndCube::stack(StackType type) {
	std::shared_ptr<AbstractStack> out;
	if (type==StackType::ISO) {
		out = std::dynamic_pointer_cast<AbstractStack, FixedLayersFromDatasetAndCubeStack>(
				std::make_shared<FixedLayersFromDatasetAndCubeStack>(this));
	}
	return out;
}


IsoSurfaceBuffer FixedLayersFromDatasetAndCube::getIsoBuffer()
{
	IsoSurfaceBuffer res;

	res.buffer  = std::make_shared<CPUImagePaletteHolder>(m_width, m_depth,ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get());
	//res.buffer = new CPUImagePaletteHolder(m_width, m_depth,ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get());
	m_currentIso->lockPointer();

	void* tab = m_currentIso->backingPointer();

	QByteArray array(m_width*m_depth*sizeof(short),0);

	memcpy(array.data(), tab,m_width*m_depth*sizeof(short) );


	m_currentIso->unlockPointer();

	res.buffer->updateTexture(array, false);


	if(m_isIsoInT)
	{
		res.originSample = 0.0f;
		res.stepSample = 1.0f;
	}
	else
	{
		res.originSample = m_sampleTransformation->b();
		res.stepSample = m_sampleTransformation->a();
	}
	return res;
}

bool FixedLayersFromDatasetAndCube::isInitialized() const {
	return m_init;
}

void FixedLayersFromDatasetAndCube::initialize() {
	if (isInitialized()) {
		return;
	}

	m_init = true;
	int index = m_currentImageIndex;
	m_currentImageIndex = -1;
	setCurrentImageIndex(index);
}






FixedLayersFromDatasetAndCubeStack::FixedLayersFromDatasetAndCubeStack(
		FixedLayersFromDatasetAndCube* data) : AbstractRangeStack(data) {
	m_data = data;
	connect(m_data, &FixedLayersFromDatasetAndCube::currentIndexChanged,
			this, &FixedLayersFromDatasetAndCubeStack::indexChangedFromData);
}

FixedLayersFromDatasetAndCubeStack::~FixedLayersFromDatasetAndCubeStack() {

}

long FixedLayersFromDatasetAndCubeStack::stackCount() const {
	return m_data->numLayers();
}

long FixedLayersFromDatasetAndCubeStack::stackIndex() const {
	return m_data->currentImageIndex();
}

QVector2D FixedLayersFromDatasetAndCubeStack::stackRange() const {
	double min = m_data->isoOrigin();
	double max = min + m_data->isoStep() * m_data->numLayers();

	double tmp = std::min(max, min);
	max = std::max(max, min);
	min = tmp;

	return QVector2D(min, max);
}

double FixedLayersFromDatasetAndCubeStack::stackStep() const {
	return std::abs(m_data->isoStep());
}

double FixedLayersFromDatasetAndCubeStack::stackValueFromIndex(long index) const {
	return m_data->isoOrigin() + m_data->isoStep() * index;
}

long FixedLayersFromDatasetAndCubeStack::stackIndexFromValue(double value) const {
	return static_cast<long>((value - m_data->isoOrigin()) / m_data->isoStep());
}

void FixedLayersFromDatasetAndCubeStack::setStackIndex(long stackIndex) {
	if (stackIndex>=0 && stackIndex<stackCount()) {
		m_data->setCurrentImageIndex(stackIndex);
	}
}

void FixedLayersFromDatasetAndCubeStack::indexChangedFromData(long stackIndex) {
	emit stackIndexChanged(stackIndex);
}



FixedLayersFromDatasetAndCube::Grid3DParameter FixedLayersFromDatasetAndCube::createGrid3DParameter(const QString& datasetPath, SeismicSurvey* survey, bool* ok)
{
	Grid3DParameter params;

	if (!datasetPath.isNull() && !datasetPath.isEmpty())
	{
		// get transforms
		SmDataset3D d3d(datasetPath.toStdString());
		AffineTransformation sampleTransfo = d3d.sampleTransfo();
		Affine2DTransformation inlineXlineTransfoForInline = d3d.inlineXlineTransfoForInline();
		Affine2DTransformation inlineXlineTransfoForXline = d3d.inlineXlineTransfoForXline();

		// params.sampleTransformation = &sampleTransfo;
		// params.ijToInlineXlineTransfoForInline = &inlineXlineTransfoForInline;
		// params.ijToInlineXlineTransfoForXline = &inlineXlineTransfoForXline;
		params.sampleTransformation = std::make_shared<AffineTransformation>(sampleTransfo);
		params.ijToInlineXlineTransfoForInline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForInline);
		params.ijToInlineXlineTransfoForXline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForXline);

		std::array<double, 6> inlineXlineTransfo =
				survey->inlineXlineToXYTransfo()->direct();
		std::array<double, 6> ijToInlineXline = d3d.inlineXlineTransfo().direct();

		std::array<double, 6> res;
		GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
				res.data());

		Affine2DTransformation ijToXYTransfo(d3d.inlineXlineTransfo().width(),
				d3d.inlineXlineTransfo().height(), res);
		// params.ijToXYTransfo = &ijToXYTransfo;
		params.ijToXYTransfo = std::make_shared<Affine2DTransformation>(ijToXYTransfo);


		inri::Xt xt(datasetPath.toStdString().c_str());
		if (xt.is_valid()) {
			params.cubeSeismicAddon.set(
					xt.startSamples(), xt.stepSamples(),
					xt.startRecord(), xt.stepRecords(),
					xt.startSlice(), xt.stepSlices());
			params.width = xt.nRecords();
			params.depth = xt.nSlices();
			params.heightFor3D = xt.nSamples();
			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(datasetPath);
			params.cubeSeismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);
		}
	}
	return params;
}

FixedLayersFromDatasetAndCube::Grid3DParameter FixedLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(const QString& horizonPath, SeismicSurvey* survey, bool* ok)
{
	Grid3DParameter params;

	if (!horizonPath.isNull() && !horizonPath.isEmpty())
	{
		// get transforms
		NvHorizonTransformGenerator hTransfoGen(horizonPath.toStdString());
		AffineTransformation sampleTransfo = hTransfoGen.sampleTransfo();
		Affine2DTransformation inlineXlineTransfoForInline = hTransfoGen.inlineXlineTransfoForInline();
		Affine2DTransformation inlineXlineTransfoForXline = hTransfoGen.inlineXlineTransfoForXline();

		params.sampleTransformation = std::make_shared<AffineTransformation>(sampleTransfo);
		params.ijToInlineXlineTransfoForInline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForInline);
		params.ijToInlineXlineTransfoForXline = std::make_shared<Affine2DTransformation>(inlineXlineTransfoForXline);

		std::array<double, 6> inlineXlineTransfo =
				survey->inlineXlineToXYTransfo()->direct();
		std::array<double, 6> ijToInlineXline = hTransfoGen.inlineXlineTransfo().direct();

		std::array<double, 6> res;
		GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
				res.data());

		Affine2DTransformation ijToXYTransfo(hTransfoGen.inlineXlineTransfo().width(),
				hTransfoGen.inlineXlineTransfo().height(), res);
		// params.ijToXYTransfo = &ijToXYTransfo;
		params.ijToXYTransfo = std::make_shared<Affine2DTransformation>(ijToXYTransfo);


		inri::Xt xt(horizonPath.toStdString().c_str());
		if (xt.is_valid()) {
			params.cubeSeismicAddon.set(
					xt.startSamples(), xt.stepSamples(),
					xt.startRecord(), xt.stepRecords(),
					xt.startSlice(), xt.stepSlices());
			params.width = xt.nSamples();
			params.depth = xt.nRecords();
			params.heightFor3D = xt.nSlices();
			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(horizonPath);
			params.cubeSeismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);
		}
	}
	return params;
}

QString FixedLayersFromDatasetAndCube::sectionToolTip() const {
	QString txt = m_sectionToolTip;

	if (txt.isNull() || txt.isEmpty()) {
		txt = name();
	}
	return txt;
}

void FixedLayersFromDatasetAndCube::setSectionToolTip(const QString& txt) {
	m_sectionToolTip = txt;
}
