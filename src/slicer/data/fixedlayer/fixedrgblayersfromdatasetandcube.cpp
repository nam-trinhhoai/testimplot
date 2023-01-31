#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedrgblayersfromdatasetandcubeimplmulti.h"
#include "fixedrgblayersfromdatasetandcubeimplmono.h"
#include "fixedrgblayersfromdatasetandcubeimpldirectories.h"

#include "fixedrgblayersfromdatasetandcubegraphicrepfactory.h"
#include "fixedattributimplfromdirectories.h"
#include "fixedattributimplfreehorizonfromdirectories.h"
#include "igraphicrepfactory.h"
#include "nvhorizontransformgenerator.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "ijkhorizon.h"
#include "affinetransformation.h"
#include "gdalloader.h"
#include "workingsetmanager.h"
#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "cudargbinterleavedimage.h"
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
#include <fileSelectorDialog.h>
#include "qmath.h"

#include <freeHorizonManager.h>
#include "surfacemesh.h"
#include "gdalloader.h"

#include <gdal_priv.h>
#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QFileDialog>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QMessageBox>
#include <QInputDialog>
#include <geotimepath.h>

#include <algorithm>
#include <chrono>
#include <memory>

// cudaBuffer need to be a float RGBD planar stack
/*FixedRGBLayersFromDatasetAndCube::FixedRGBLayersFromDatasetAndCube(QString cube,
			QString name, WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent) : IData(workingSet, parent) {
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

	m_repFactory.reset(new FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(this));

	m_currentIso.reset(new CUDAImagePaletteHolder(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));
	m_currentRGB.reset(new CUDARGBImage(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));

	loadIsoAndRgb(cube);

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
	m_simplifyMeshSteps = 10;

	setCurrentImageIndex(0);
}

FixedRGBLayersFromDatasetAndCube::FixedRGBLayersFromDatasetAndCube(QString rgb2Path, QString rgb1Path, QString name,
		WorkingSetManager *workingSet, const Grid3DParameter& params,
		QObject *parent) : IData(workingSet, parent) {
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

	m_repFactory.reset(new FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(this));

	m_currentIso.reset(new CUDAImagePaletteHolder(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));
	m_currentRGB.reset(new CUDARGBImage(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));

	loadIsoAndRgb(rgb2Path, rgb1Path);

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

	m_simplifyMeshSteps = 10;
	m_compressionMesh=0;
	setCurrentImageIndex(0);
}*/

FixedRGBLayersFromDatasetAndCube::FixedRGBLayersFromDatasetAndCube(
		QString name, WorkingSetManager *workingSet,
		const Grid3DParameter& params, QObject *parent) : IData(workingSet, parent) {
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

	m_repFactory.reset(new FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(this));

	m_currentIso.reset(new CPUImagePaletteHolder(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));
	m_currentRGB.reset(new CUDARGBInterleavedImage(m_width, m_depth,
			ImageFormats::QSampleType::INT16, m_ijToXYTransfo.get(),
			this));

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

	m_timerRefresh = new QTimer();
	connect(m_timerRefresh,SIGNAL(timeout()),this,SLOT(nextCurrentIndex()));

}

QString FixedRGBLayersFromDatasetAndCube::getCurrentObjFile() const
{
	return getObjFile(m_currentImageIndex);
}

SurfaceMeshCache* FixedRGBLayersFromDatasetAndCube::getMeshCache(int newIndex)
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

bool FixedRGBLayersFromDatasetAndCube::isIndexCache(int newIndex)
{
	return (m_mode==CACHE && ((newIndex-m_cacheFirstIndex)%m_cacheStepIndex)==0 && ((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)>=0 &&
			((newIndex-m_cacheFirstIndex)/m_cacheStepIndex)<=((m_cacheLastIndex-m_cacheFirstIndex)/m_cacheStepIndex));
	//return (index >0 && index < m_numLayers);
}

int FixedRGBLayersFromDatasetAndCube::getSimplifyMeshSteps()const
{
	return m_simplifyMeshSteps;
}
void FixedRGBLayersFromDatasetAndCube::setSimplifyMeshSteps(int steps)
{
	if(m_simplifyMeshSteps  != steps)
	{
		m_simplifyMeshSteps  = steps;
		emit simplifyMeshStepsChanged(steps);
	}
}



int FixedRGBLayersFromDatasetAndCube::getCompressionMesh()const
{
	return m_compressionMesh;
}

void FixedRGBLayersFromDatasetAndCube::setCompressionMesh(int compress)
{
	if(m_compressionMesh  != compress)
	{
		m_compressionMesh  = compress;
		emit compressionMeshChanged(compress);
	}
}

FixedRGBLayersFromDatasetAndCube::~FixedRGBLayersFromDatasetAndCube() {
	//	if (m_fRgb2!=nullptr) {
	//		fclose(m_fRgb2);
	//	}
	//	if (m_fRgb1!=nullptr) {
	//		fclose(m_fRgb1);
	//	}
}

//void FixedRGBLayersFromDatasetAndCube::loadIsoAndRgb(QString rgb2, QString rgb1) {
//	m_fRgb2 = fopen(rgb2.toStdString().c_str(), "r");
//	m_rgb2Path = rgb2;
//	std::size_t sz;
//	bool isValid;
//	if (m_fRgb2!=nullptr) {
//		fseek(m_fRgb2, 0L, SEEK_END);
//		sz = ftell(m_fRgb2);
//		fseek(m_fRgb2, 0L, SEEK_SET);
//		isValid = true;
//	}
//
//	if (isValid) {
//		m_numLayers = sz / (width() * depth() * sizeof(short) * 4 );
//
//		m_fRgb1 = fopen(rgb1.toStdString().c_str(), "r");
//		if (m_fRgb1!=nullptr) {
//			m_rgb1Path = rgb1;
//			std::size_t sz1;
//			fseek(m_fRgb1, 0L, SEEK_END);
//			sz = ftell(m_fRgb1);
//			fseek(m_fRgb1, 0L, SEEK_SET);
//			m_useRgb1 = sz==m_numLayers*width() * depth() * sizeof(char) * 3;
//			if (!m_useRgb1) {
//				fclose(m_fRgb1);
//			}
//		} else {
//			m_fRgb1 = nullptr;
//			m_useRgb1 = false;
//		}
//	} else {
//		m_numLayers = 0;
//		m_fRgb1 = nullptr;
//		m_useRgb1 = false;
//	}
//	m_isoOrigin = (m_numLayers-1) * (-m_isoStep);
//}

unsigned int FixedRGBLayersFromDatasetAndCube::width() const {
	return m_width;
}

unsigned int FixedRGBLayersFromDatasetAndCube::depth() const {
	return m_depth;
}

unsigned int FixedRGBLayersFromDatasetAndCube::getNbProfiles() const {
	return depth();
}

unsigned int FixedRGBLayersFromDatasetAndCube::getNbTraces() const {
	return width();
}

unsigned int FixedRGBLayersFromDatasetAndCube::heightFor3D() const { // only use for 3d
	return m_heightFor3D;
}

float FixedRGBLayersFromDatasetAndCube::getStepSample() {
	return m_sampleTransformation->a();
}

float FixedRGBLayersFromDatasetAndCube::getOriginSample() {
	return m_sampleTransformation->b();
}

//IData
IGraphicRepFactory* FixedRGBLayersFromDatasetAndCube::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid FixedRGBLayersFromDatasetAndCube::dataID() const {
	return m_uuid;
}

QString FixedRGBLayersFromDatasetAndCube::name() const {
	return m_name;
}

std::size_t FixedRGBLayersFromDatasetAndCube::numLayers() {
	return m_numLayers;
}


const QList<QString>& FixedRGBLayersFromDatasetAndCube::layers() const {
	return m_layers;
}

long FixedRGBLayersFromDatasetAndCube::currentImageIndex() const {
	return m_currentImageIndex;
}

void FixedRGBLayersFromDatasetAndCube::nextCurrentIndex()
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


void FixedRGBLayersFromDatasetAndCube::play(int interval,int coef,bool looping)
{
	m_modePlay = !m_modePlay;
	m_loop = looping;
	m_coef = coef;

	if(m_modePlay)
	{
		//int indexmin = qMin (m_cacheFirstIndex,m_cacheLastIndex);
		//if (m_mode==CACHE) setCurrentImageIndex(indexmin);
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

void FixedRGBLayersFromDatasetAndCube::setCurrentImageIndex(long newIndex) {
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

std::vector<FixedRGBLayersFromDatasetAndCube*> FixedRGBLayersFromDatasetAndCube::createDataFromDataset(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
				QString dataType,
				QObject *parent) {

	QString seachDir = survey->idPath() + "/ImportExport/IJK/";
	std::vector<FixedRGBLayersFromDatasetAndCube*> out;

	std::vector<std::shared_ptr<FixedAttributImplFromDirectories::Parameters>> directoriesParams;
	directoriesParams = FixedAttributImplFromDirectories::findPotentialData(seachDir);

	for (int i=0; i<directoriesParams.size(); i++)
	{
		bool isValid = true;
		QString sismageName = directoriesParams[i]->sismageName(&isValid);
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
		Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
		FixedRGBLayersFromDatasetAndCube* outObj = directoriesParams[i]->create(sismageName, workingSet, params, parent);
		out.push_back(outObj);
		qDebug() << "**  " << sismageName << " " << datasetPath;
	}
	return out;
}

std::vector<FixedRGBLayersFromDatasetAndCube*> FixedRGBLayersFromDatasetAndCube::createHorizonIsoDataFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
				QString dataType,
				QObject *parent) {

	QString seachDir = survey->idPath() + "/ImportExport/IJK/HORIZONS/ISOVAL/";
	qDebug() << seachDir;

	std::vector<FixedRGBLayersFromDatasetAndCube*> out;

	std::vector<std::shared_ptr<FixedAttributImplFromDirectories::Parameters>> directoriesParams;
	directoriesParams = FixedAttributImplFromDirectories::findPotentialData(seachDir);

	bool isValid = directoriesParams.size()>0;
	bool errorLogging = false;
	if ( !isValid ) return out;
	int index = 0;
	std::vector<int> idx;
	if ( isValid )
	{
		std::vector<QString> displayNames;
		for (long i=0; i<directoriesParams.size(); i++) {
			displayNames.push_back(directoriesParams[i]->name());
		}
		FileSelectorDialog dialog(&displayNames, "select data");
		dialog.setMultipleSelection(true);
		dialog.setMainSearchType(FileSelectorDialog::all);
		int result = dialog.exec();
		if ( result == QDialog::Accepted )
		{
			idx = dialog.getMultipleSelectedIndex();
			isValid == idx.size() > 0;
		}
		/*
			StringSelectorDialog dialog(&displayNames, "select data");
			int result = dialog.exec();
			index = dialog.getSelectedIndex();
			isValid = result == QDialog::Accepted && index < directoriesParams.size() && index >= 0;
		 */
	}
	else
	{
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB1 data");
		errorLogging = true;
	}
	QString sismageName;
	if ( isValid )
	{
		for (int n=0; n<idx.size(); n++)
		{
			bool isValid2 = true;
			sismageName = directoriesParams[idx[n]]->sismageName(&isValid);
			if ( isValid2 )
			{
				// QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
				QString datasetPath = survey->idPath() + "/DATA/SEISMIC/" + sismageName + ".xt";
				if (!datasetPath.isNull() && !datasetPath.isEmpty())
				{
					Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
					QString suffix;
					if (useRgb1) {
						suffix = " (rgb1)";
					} else {
						suffix = " (rgb2)";
					}
					if (isValid) {
						out.push_back(directoriesParams[idx[n]]->create(sismageName, workingSet, params, parent));
					}
				}
				else { }
			}
			else
			{

			}
		}
	}
	return out;
}


std::vector<FixedRGBLayersFromDatasetAndCube*> FixedRGBLayersFromDatasetAndCube::createDataFreeHorizonFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
				QString dataType,
				QObject *parent) {

	// QString seachDir = survey->idPath() + "/ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/";
	QString seachDir = survey->idPath() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
	std::vector<FixedRGBLayersFromDatasetAndCube*> out;

	std::vector<std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters>> directoriesParams;
	directoriesParams = FixedAttributImplFreeHorizonFromDirectories::findPotentialData(seachDir);

	bool isValid = directoriesParams.size()>0;
	bool errorLogging = false;
	if ( !isValid ) return out;
	int index = 0;
	std::vector<int> idx;
	if ( isValid )
	{
		std::vector<QString> displayNames;
		for (long i=0; i<directoriesParams.size(); i++) {
			displayNames.push_back(directoriesParams[i]->name());
		}
		FileSelectorDialog dialog(&displayNames, "select data");
		dialog.setMultipleSelection(true);
		dialog.setMainSearchType(FileSelectorDialog::all);
		int result = dialog.exec();
		if ( result == QDialog::Accepted )
		{
			idx = dialog.getMultipleSelectedIndex();
			isValid == idx.size() > 0;
		}
		/*
		StringSelectorDialog dialog(&displayNames, "select data");
		int result = dialog.exec();
		index = dialog.getSelectedIndex();
		isValid = result == QDialog::Accepted && index < directoriesParams.size() && index >= 0;
		*/
	}
	else
	{
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB1 data");
		errorLogging = true;
	}

	QString sismageName;
	if ( isValid )
	{
		for (int n=0; n<idx.size(); n++)
		{
			bool isValid2 = true;
			sismageName = directoriesParams[idx[n]]->sismageName(&isValid2);
			if ( isValid2 )
			{
				// QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
				QString datasetPath = survey->idPath() + "/DATA/SEISMIC/" + sismageName + ".xt";
				if (!datasetPath.isNull() && !datasetPath.isEmpty())
				{
					Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
					if (isValid)
					{
						out.push_back(directoriesParams[idx[n]]->create(sismageName, workingSet, params, parent));
					}
				}
			}
		}
	}

	/*
	for (int i=0; i<directoriesParams.size(); i++)
	{
		bool isValid = true;
		QString sismageName = directoriesParams[i]->sismageName(&isValid);
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
		Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
		FixedRGBLayersFromDatasetAndCube* outObj = directoriesParams[i]->create(sismageName, workingSet, params, parent);
		out.push_back(outObj);
		qDebug() << "**  " << sismageName << " " << datasetPath;
	}
	*/
	return out;
}



std::vector<FixedRGBLayersFromDatasetAndCube*> FixedRGBLayersFromDatasetAndCube::createDataFreeHorizonFromDataset(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
				QString dataType,
				QObject *parent) {

	QString seachDir = survey->idPath() + "/ImportExport/IJK/";
	std::vector<FixedRGBLayersFromDatasetAndCube*> out;

	std::vector<std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters>> directoriesParams;
	directoriesParams = FixedAttributImplFreeHorizonFromDirectories::findPotentialData(seachDir);

	for (int i=0; i<directoriesParams.size(); i++)
	{
		bool isValid = true;
		QString sismageName = directoriesParams[i]->sismageName(&isValid);
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
		Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
		FixedRGBLayersFromDatasetAndCube* outObj = directoriesParams[i]->create(sismageName, workingSet, params, parent);
		out.push_back(outObj);
		qDebug() << "**  " << sismageName << " " << datasetPath;
	}
	return out;
}


FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCube::createDataFreeHorizonFromHorizonPath(QString horizonPath,
		WorkingSetManager *workingSet,
		SeismicSurvey* survey,
		QObject *parent) {

	FixedRGBLayersFromDatasetAndCube* out = nullptr;

	std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters> param0;
	param0 = FixedAttributImplFreeHorizonFromDirectories::findData(horizonPath);
	bool isValid = true;
	QString sismageName = param0->sismageName(&isValid);

// 	QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
	// QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
	QString datasetPath = survey->idPath() + "/DATA/SEISMIC/" + sismageName + ".xt";
	// QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
// 	QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
	Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
	out = param0->create(sismageName, workingSet, params, parent);

/*
	for (int i=0; i<directoriesParams.size(); i++)
	{
		bool isValid = true;
		QString sismageName = directoriesParams[i]->sismageName(&isValid);
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);
		Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);
		FixedRGBLayersFromDatasetAndCube* outObj = directoriesParams[i]->create(sismageName, workingSet, params, parent);
		out.push_back(outObj);
		qDebug() << "**  " << sismageName << " " << datasetPath;
	}
	*/
	return out;
}


FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUI(QString prefix,
		WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
		QString dataType,
		QObject *parent) {
	qDebug() << dataType;
	// m_dataType = dataType;
	QString seachDir = survey->idPath() + "/ImportExport/IJK/";
	std::vector<std::shared_ptr<AbstractConstructorParams>> constructorParams;
	if (QDir(seachDir).exists()) {
		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
		for (const QFileInfo& fileInfo : infoList) {
			QDir dir(fileInfo.absoluteFilePath());
			if(dir.cd("cubeRgt2RGB")) {
				std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMulti::Parameters>> multiParams;
				std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplMono::Parameters>> monoParams;
				std::vector<std::shared_ptr<FixedRGBLayersFromDatasetAndCubeImplDirectories::Parameters>> directoriesParams;

				if (useRgb1) {
					multiParams = FixedRGBLayersFromDatasetAndCubeImplMulti::findPotentialDataRgb1(dir.absolutePath());
				} else {
					multiParams = FixedRGBLayersFromDatasetAndCubeImplMulti::findPotentialDataRgb2(dir.absolutePath());
				}
				if (useRgb1) {
					monoParams = FixedRGBLayersFromDatasetAndCubeImplMono::findPotentialDataRgb1(dir.absolutePath());
				} else {
					monoParams = FixedRGBLayersFromDatasetAndCubeImplMono::findPotentialDataRgb2(dir.absolutePath());
				}
				if ( !useRgb1 )
				{
					directoriesParams = FixedRGBLayersFromDatasetAndCubeImplDirectories::findPotentialDataRgb2(dir.absolutePath(), dataType);
				}

				for (int i=0; i<multiParams.size(); i++) {
					if (multiParams[i]->rgb1Valid() || !useRgb1) {
						constructorParams.push_back(std::dynamic_pointer_cast<AbstractConstructorParams>(multiParams[i]));
					}
				}
				for (int i=0; i<monoParams.size(); i++) {
					if (monoParams[i]->rgb1Valid() || !useRgb1) {
						constructorParams.push_back(std::dynamic_pointer_cast<AbstractConstructorParams>(monoParams[i]));
					}
				}

				for (int i=0; i<directoriesParams.size(); i++) {
					constructorParams.push_back(std::dynamic_pointer_cast<AbstractConstructorParams>(directoriesParams[i]));
				}
			}
		}
	}

	bool isValid = constructorParams.size()>0;
	bool errorLogging = false;
	int rgb1Index = 0;
	if (isValid) {
		QStringList rgb1NamesBuf;
		for (long i=0; i<constructorParams.size(); i++) {
			rgb1NamesBuf << constructorParams[i]->name();
		}
		QString title;
		if (useRgb1) {
			title = "Select RGB1";
		} else {
			title = "Select RGB2";
		}
		StringSelectorDialog dialog(&rgb1NamesBuf, title);
		int result = dialog.exec();
		rgb1Index = dialog.getSelectedIndex();

		isValid = result==QDialog::Accepted && rgb1Index<constructorParams.size() && rgb1Index>=0;
	} else {
		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any RGB1 data");
		errorLogging = true;
	}

	FixedRGBLayersFromDatasetAndCube* outObj = nullptr;
	QString sismageName;
	if (isValid) {
		sismageName = constructorParams[rgb1Index]->sismageName(&isValid);
	}
	if (isValid) {
		QString datasetPath = DatasetRelatedStorageImpl::getDatasetPath(survey->idPath(), sismageName);

		if (!datasetPath.isNull() && !datasetPath.isEmpty()) {
			Grid3DParameter params = createGrid3DParameter(datasetPath, survey, &isValid);

			QString suffix;
			if (useRgb1) {
				suffix = " (rgb1)";
			} else {
				suffix = " (rgb2)";
			}
			if (isValid) {
				outObj = constructorParams[rgb1Index]->create(prefix+sismageName+suffix, workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}

FixedRGBLayersFromDatasetAndCube::Grid3DParameter FixedRGBLayersFromDatasetAndCube::createGrid3DParameter(
		const QString& datasetPath, SeismicSurvey* survey, bool* ok) {
	Grid3DParameter params;
	*ok = true;

	inri::Xt xt(datasetPath.toStdString().c_str());
	if (xt.is_valid()) {// check if xt valid before using SmDataset3D because it use xt class but does not check validity
		params.cubeSeismicAddon.set(
				xt.startSamples(), xt.stepSamples(),
				xt.startRecord(), xt.stepRecords(),
				xt.startSlice(), xt.stepSlices());
		params.width = xt.nRecords();
		params.depth = xt.nSlices();
		params.heightFor3D = xt.nSamples();
		int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(datasetPath);
		params.cubeSeismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);
		*ok = timeOrDepth==0 || timeOrDepth==1;
	} else {
		std::cerr << "xt cube is not valid (" << datasetPath.toStdString() << ")" << std::endl;
		*ok = false;
	}

	if (*ok) {
		// get transforms
		SmDataset3D d3d(datasetPath.toStdString());
		AffineTransformation sampleTransfo = d3d.sampleTransfo();
		Affine2DTransformation inlineXlineTransfoForInline = d3d.inlineXlineTransfoForInline();
		Affine2DTransformation inlineXlineTransfoForXline = d3d.inlineXlineTransfoForXline();

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
		params.ijToXYTransfo = std::make_shared<Affine2DTransformation>(ijToXYTransfo);
	}

	return params;
}

FixedRGBLayersFromDatasetAndCube::Grid3DParameter FixedRGBLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(const QString& horizonPath, SeismicSurvey* survey, bool* ok)
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

QString FixedRGBLayersFromDatasetAndCube::extractListAndJoin(QStringList list, long beg, long end, QString joinStr) {
	QStringList newList;
	for (long i=beg; i<=end; i++) {
		newList << list[i];
	}
	return newList.join(joinStr);
}

/*
FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUIRgb1(QString prefix,
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
			Grid3DParameter params;

			// get transforms
			SmDataset3D d3d(datasetPath.toStdString());
			AffineTransformation sampleTransfo = d3d.sampleTransfo();
			Affine2DTransformation inlineXlineTransfoForInline = d3d.inlineXlineTransfoForInline();
			Affine2DTransformation inlineXlineTransfoForXline = d3d.inlineXlineTransfoForXline();

			params.sampleTransformation = &sampleTransfo;
			params.ijToInlineXlineTransfoForInline = &inlineXlineTransfoForInline;
			params.ijToInlineXlineTransfoForXline = &inlineXlineTransfoForXline;
			std::array<double, 6> inlineXlineTransfo =
					survey->inlineXlineToXYTransfo()->direct();
			std::array<double, 6> ijToInlineXline = d3d.inlineXlineTransfo().direct();

			std::array<double, 6> res;
			GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
					res.data());

			Affine2DTransformation ijToXYTransfo(d3d.inlineXlineTransfo().width(),
					d3d.inlineXlineTransfo().height(), res);
			params.ijToXYTransfo = &ijToXYTransfo;

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
				outObj = new FixedRGBLayersFromDatasetAndCube(
							cube2File, cube1File, prefix+sismageName+" (rgb1)", workingSet, params, parent);
			}
		} else {

		}
	} else {
		// no message to put because qinputdialog gave invalid input, that happen only if the user choose to.
		errorLogging = true;
	}

	return outObj;
}*/

FixedRGBLayersFromDatasetAndCube::Mode FixedRGBLayersFromDatasetAndCube::mode() const {
	return m_mode;
}

void FixedRGBLayersFromDatasetAndCube::moveToReadMode() {
	if (m_mode != READ) {
		QMutexLocker locker(&m_lockRead);
		m_mode = READ;
		m_cacheFirstIso = 0;
		m_cacheLastIso = 0;
		m_cacheStepIso = 1;
		m_cacheList.clear();
	}
	emit modeChanged();
}

bool FixedRGBLayersFromDatasetAndCube::moveToCacheMode(long firstIso, long lastIso, long isoStep) {
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
				loading = getImageForIndex(index, cache.rgb, cache.iso);
				if (!loading) {
					break;
				}
				if (m_useRgb1) {
					cache.redRange = QVector2D(0, 255); // because of rgb1
					cache.greenRange = QVector2D(0, 255); // because of rgb1
					cache.blueRange = QVector2D(0, 255); // because of rgb1
					QHistogram emptyHisto;
					emptyHisto.setRange(cache.redRange);
					cache.redHistogram = emptyHisto;
					cache.greenHistogram = emptyHisto;
					cache.blueHistogram = emptyHisto;
				} else {
					const short* oriData = static_cast<const short*>(static_cast<const void*>(cache.rgb.constData()));
					std::vector<short> rgbPlanar;
					rgbPlanar.resize(mapSize*3);
					for (std::size_t pixelIdx=0; pixelIdx<mapSize; pixelIdx++) {
						rgbPlanar[pixelIdx] = oriData[pixelIdx*3];
						rgbPlanar[pixelIdx + mapSize] = oriData[pixelIdx*3+1];
						rgbPlanar[pixelIdx + mapSize*2] = oriData[pixelIdx*3+2];
					}
					paletteHolder.updateTexture(rgbPlanar.data(), false);
					cache.redRange = paletteHolder.dataRange();
					cache.redHistogram = paletteHolder.computeHistogram(paletteHolder.dataRange(), QHistogram::HISTOGRAM_SIZE);
					paletteHolder.updateTexture(rgbPlanar.data() + mapSize, false);
					cache.greenRange = paletteHolder.dataRange();
					cache.greenHistogram = paletteHolder.computeHistogram(paletteHolder.dataRange(), QHistogram::HISTOGRAM_SIZE);
					paletteHolder.updateTexture(rgbPlanar.data() + mapSize*2, false);
					cache.blueRange = paletteHolder.dataRange();
					cache.blueHistogram = paletteHolder.computeHistogram(paletteHolder.dataRange(), QHistogram::HISTOGRAM_SIZE);
				}
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
				const short* isoBuf = static_cast<const short*>(static_cast<const void*>(cache.iso.constData()));

				SurfaceMesh::createCache<short>(cache.meshCache,isoBuf,m_width,m_depth,m_simplifyMeshSteps,origin,scal,getObjFile(index),ijToXYTranformSwapped,m_compressionMesh);
				m_cacheList.push_back(cache);
				emit valueProgressBarChanged(index * indexInc);
				QCoreApplication::processEvents();
			}
			emit endProgressBar();

			if (loading) {
				QMutexLocker locker(&m_lockRead);
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

long FixedRGBLayersFromDatasetAndCube::cacheFirstIndex() const {
	return m_cacheFirstIndex;
}

long FixedRGBLayersFromDatasetAndCube::cacheLastIndex() const {
	return m_cacheLastIndex;
}

long FixedRGBLayersFromDatasetAndCube::cacheStepIndex() const {
	return m_cacheStepIndex;
}

bool FixedRGBLayersFromDatasetAndCube::useRgb1() const {
	return m_useRgb1;
}

//QString FixedRGBLayersFromDatasetAndCube::rgb2Path() const {
//	return m_rgb2Path;
//}
//
//QString FixedRGBLayersFromDatasetAndCube::rgb1Path() const {
//	return m_rgb1Path;
//}

CubeSeismicAddon FixedRGBLayersFromDatasetAndCube::cubeSeismicAddon() const {
	return m_cubeSeismicAddon;
}

const AffineTransformation* FixedRGBLayersFromDatasetAndCube::sampleTransformation() const {
	return m_sampleTransformation.get();
}

const Affine2DTransformation* FixedRGBLayersFromDatasetAndCube::ijToXYTransfo() const {
	return m_ijToXYTransfo.get();
}

const Affine2DTransformation* FixedRGBLayersFromDatasetAndCube::ijToInlineXlineTransfoForInline() const {
	return m_ijToInlineXlineTransfoForInline.get();
}

const Affine2DTransformation* FixedRGBLayersFromDatasetAndCube::ijToInlineXlineTransfoForXline() const {
	return m_ijToInlineXlineTransfoForXline.get();
}

std::vector<StackType> FixedRGBLayersFromDatasetAndCube::stackTypes() const {
	std::vector<StackType> typeVect;
	typeVect.push_back(StackType::ISO);
	return typeVect;
}

std::shared_ptr<AbstractStack> FixedRGBLayersFromDatasetAndCube::stack(StackType type) {
	std::shared_ptr<AbstractStack> out;
	if (type==StackType::ISO) {
		out = std::dynamic_pointer_cast<AbstractStack, FixedRGBLayersFromDatasetAndCubeStack>(
				std::make_shared<FixedRGBLayersFromDatasetAndCubeStack>(this));
	}
	return out;
}

/*void FixedRGBLayersFromDatasetAndCube::copyGDALBufToFloatBufInterleaved(void* oriBuf, float* outBuf,
		std::size_t width, std::size_t height, std::size_t numBands, std::size_t offset,
		ImageFormats::QColorFormat colorFormat, ImageFormats::QSampleType sampleType,
		GDALRasterBand* hBand) {
	if (sampleType==ImageFormats::QSampleType::UINT8) {
		_copyGDALBufToFloatBufInterleaved<unsigned char>(static_cast<unsigned char*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::UINT16) {
		_copyGDALBufToFloatBufInterleaved<unsigned short>(static_cast<unsigned short*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::INT16) {
		_copyGDALBufToFloatBufInterleaved<short>(static_cast<short*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::UINT32) {
		_copyGDALBufToFloatBufInterleaved<unsigned int>(static_cast<unsigned int*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::INT32) {
		_copyGDALBufToFloatBufInterleaved<int>(static_cast<int*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::FLOAT32) {
		_copyGDALBufToFloatBufInterleaved<float>(static_cast<float*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	}
}*/

long long FixedRGBLayersFromDatasetAndCube::cacheLayerMemoryCost() const {
	long long longWidth = m_width;
	long long longDepth = m_depth;
	// iso + rgb as short
	long long cost = 8 * longWidth * longDepth;

	long long reducedWidth = (longWidth - 1) / m_defaultSimplifyMeshSteps + 1;
	long long reducedHeight = (longWidth - 1) / m_defaultSimplifyMeshSteps + 1;

	// vertex xyz + normal xyz as float + texture uv as unsigned int
	cost += 4 * 8 * reducedWidth * reducedHeight;

	// triangles p1 p2 and p3 as unsigned int
	cost += 4 * 6 * (reducedWidth - 1) * (reducedHeight - 1);
	return cost;
}

void FixedRGBLayersFromDatasetAndCube::initLayersList() {
	m_layers.clear();
	for (std::size_t i=0; i<m_numLayers; i++) {
		// fprintf(stderr, "---> %d %d\n", i, m_numLayers);
		m_layers.push_back(QString::number(i*m_isoStep));
	}
}

bool FixedRGBLayersFromDatasetAndCube::isMinimumValueActive() const {
	return m_useMinimumValue;
}

void FixedRGBLayersFromDatasetAndCube::setMinimumValueActive(bool active) {
	if (m_useMinimumValue!=active) {
		m_useMinimumValue = active;
		emit minimumValueActivated(m_useMinimumValue);
	}
}

float FixedRGBLayersFromDatasetAndCube::minimumValue() const {
	return m_minimumValue;
}

void FixedRGBLayersFromDatasetAndCube::setMinimumValue(float minimumValue) {
	if (m_minimumValue!=minimumValue) {
		m_minimumValue = minimumValue;
		emit minimumValueChanged(m_minimumValue);
	}
}

bool FixedRGBLayersFromDatasetAndCube::isInitialized() const {
	return m_init;
}

void FixedRGBLayersFromDatasetAndCube::initialize() {
	if (isInitialized()) {
		return;
	}

	m_init = true;
	int index = m_currentImageIndex;
	m_currentImageIndex = -1;
	setCurrentImageIndex(index);
}

FixedRGBLayersFromDatasetAndCubeStack::FixedRGBLayersFromDatasetAndCubeStack(
		FixedRGBLayersFromDatasetAndCube* data) : AbstractRangeStack(data) {
	m_data = data;
	connect(m_data, &FixedRGBLayersFromDatasetAndCube::currentIndexChanged,
			this, &FixedRGBLayersFromDatasetAndCubeStack::indexChangedFromData);
}

FixedRGBLayersFromDatasetAndCubeStack::~FixedRGBLayersFromDatasetAndCubeStack() {

}

long FixedRGBLayersFromDatasetAndCubeStack::stackCount() const {
	return m_data->numLayers();
}

long FixedRGBLayersFromDatasetAndCubeStack::stackIndex() const {
	return m_data->currentImageIndex();
}

QVector2D FixedRGBLayersFromDatasetAndCubeStack::stackRange() const {
	double min = m_data->isoOrigin();
	double max = min + m_data->isoStep() * m_data->numLayers();

	double tmp = std::min(max, min);
	max = std::max(max, min);
	min = tmp;

	return QVector2D(min, max);
}

double FixedRGBLayersFromDatasetAndCubeStack::stackStep() const {
	return std::abs(m_data->isoStep());
}

double FixedRGBLayersFromDatasetAndCubeStack::stackValueFromIndex(long index) const {
	return m_data->isoOrigin() + m_data->isoStep() * index;
}

long FixedRGBLayersFromDatasetAndCubeStack::stackIndexFromValue(double value) const {
	return static_cast<long>((value - m_data->isoOrigin()) / m_data->isoStep());
}

void FixedRGBLayersFromDatasetAndCubeStack::setStackIndex(long stackIndex) {
	if (stackIndex>=0 && stackIndex<stackCount()) {
		m_data->setCurrentImageIndex(stackIndex);
	}
}

void FixedRGBLayersFromDatasetAndCubeStack::indexChangedFromData(long stackIndex) {
	emit stackIndexChanged(stackIndex);
}

FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams::AbstractConstructorParams(
		QString name, bool rgb1Valid) {
	m_name = name;
	m_rgb1Valid = rgb1Valid;
}

FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams::~AbstractConstructorParams() {

}

QString FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams::name() const {
	return m_name;
}

bool FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams::rgb1Valid() const {
	return m_rgb1Valid;
}

IsoSurfaceBuffer FixedRGBLayersFromDatasetAndCube::getIsoBuffer()
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



