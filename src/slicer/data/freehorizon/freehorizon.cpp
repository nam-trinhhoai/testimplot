
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include "icontreewidgetitemdecorator.h"
#include "seismicsurvey.h"
#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>
#include "freehorizongraphicrepfactory.h"
#include <seismic3ddataset.h>
#include <freehorizonattribut.h>
#include "freehorizon.h"
// #include "markergraphicrepfactory.h"
// #include "wellpick.h"
// #include "wellbore.h"
// #include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "exportnvhorizondialog.h"
#include <freeHorizonManager.h>
#include <fixedlayerimplfreehorizonfromdataset.h>
#include <fixedattributimplfreehorizonfromdirectories.h>
#include "rgblayerimplfreehorizonslice.h"
#include "LayerSlice.h"
#include "rgblayerfreehorizongraphicrepfactory.h"
#include "cudargbimage.h"
#include "omp.h"
#include <fixedlayerimplfreehorizonfromdatasetandcube.h>


FreeHorizon::FreeHorizon(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent) :
		IData(workingSet, parent), m_name(name) {
	m_uuid = QUuid::createUuid();
	m_path = path;
	bool colorOk = false;
	QColor color = FreeHorizonQManager::loadColorFromPath(m_path, &colorOk);
	if ( colorOk )
	{
		m_color = color;
	}
	else
	{
		m_color = Qt::white;
	}
	FreeHorizonManager::PARAM horizonParam = FreeHorizonManager::dataSetGetParam(m_path.toStdString()+"/"+FreeHorizonManager::isoDataName);
	if (horizonParam.axis==inri::Xt::Time)
	{
		m_sampleUnit = SampleUnit::TIME;
	}
	else if (horizonParam.axis==inri::Xt::Depth)
	{
		m_sampleUnit = SampleUnit::DEPTH;
	}
	else
	{
		m_sampleUnit = SampleUnit::NONE;
	}

	m_parent = parent;
	m_workingSet = workingSet;
	m_decorator = nullptr;
	m_repFactory = new FreeHorizonGraphicRepFactory(this);
	m_attribut.clear();
	m_survey = survey;
	freeHorizonAttributCreate();

}

FreeHorizon::~FreeHorizon() {
	if (!m_isoData.isNull()) {
		disconnect(m_isoData.data(), &FixedLayerImplFreeHorizonFromDatasetAndCube::colorChanged,
								this, &FreeHorizon::setColor);
	}
}

const AffineTransformation* FreeHorizon::sampleTransformation() const
{
	if(m_attribut.size()>0) return m_attribut[0].sampleTransformation();
	return  new AffineTransformation();
}

CubeSeismicAddon FreeHorizon::cubeSeismicAddon() const
{
	if(m_attribut.size()>0) return m_attribut[0].cubeSeismicAddon();
	return CubeSeismicAddon();
}

const Affine2DTransformation* FreeHorizon::ijToXYTransfo(QString name)const
{
	for(int i=0;i<m_attribut.size();i++)
	{
		if(m_attribut[i].name() == name) return m_attribut[i].ijToXYTransfo();
	}
	return nullptr;//new Affine2DTransformation();
}

FreeHorizon::Attribut FreeHorizon::getLayer(QString name)
{
	for(int i=0;i<m_attribut.size();i++)
		{
			if(m_attribut[i].name() == name)
			{

				return m_attribut[i];
			}
		}
	Attribut attribut;
	return attribut;
}

QString FreeHorizon::Attribut::name() const
{
	if(getData() == nullptr) return "";
	return getData()->name();
}

IData* FreeHorizon::Attribut::getData() const
{
	IData* data = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		data = pFixedRGBLayersFromDatasetAndCube;
	}
	else if (pFixedLayerFromDataset)
	{
		data = pFixedLayerFromDataset;
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		data = pRGBLayerImplFreeHorizonOnSlice;
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		data = pFixedLayersImplFreeHorizonFromDatasetAndCube;
	}
	return data;
}

const AffineTransformation* FreeHorizon::Attribut::sampleTransformation() const
{
	const AffineTransformation* transform = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		transform = pFixedRGBLayersFromDatasetAndCube->sampleTransformation();
	}
	else if (pFixedLayerFromDataset)
	{
		transform = pFixedLayerFromDataset->dataset()->sampleTransformation();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		transform = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->sampleTransformation();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		transform = pFixedLayersImplFreeHorizonFromDatasetAndCube->sampleTransformation();
	}
	return transform;
}
CubeSeismicAddon FreeHorizon::Attribut::cubeSeismicAddon() const {
	CubeSeismicAddon addon;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		addon = pFixedRGBLayersFromDatasetAndCube->cubeSeismicAddon();
	}
	else if (pFixedLayerFromDataset)
	{
		addon = pFixedLayerFromDataset->dataset()->cubeSeismicAddon();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		addon = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->cubeSeismicAddon();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		addon = pFixedLayersImplFreeHorizonFromDatasetAndCube->cubeSeismicAddon();
	}
	return addon;
}

const Affine2DTransformation* FreeHorizon::Attribut::ijToXYTransfo() const
{
	const Affine2DTransformation* transform = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		transform = pFixedRGBLayersFromDatasetAndCube->ijToXYTransfo();
	}
	else if (pFixedLayerFromDataset)
	{
		transform = pFixedLayerFromDataset->dataset()->ijToXYTransfo();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		transform = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->ijToXYTransfo();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		transform = pFixedLayersImplFreeHorizonFromDatasetAndCube->ijToXYTransfo();
	}
	return transform;
}


const Affine2DTransformation* FreeHorizon::Attribut::ijToInlineXlineTransfoForInline() const
{
	const Affine2DTransformation* transform = nullptr;
		if (pFixedRGBLayersFromDatasetAndCube)
		{
			transform = pFixedRGBLayersFromDatasetAndCube->ijToInlineXlineTransfoForInline();
		}
		else if (pFixedLayerFromDataset)
		{
			transform = pFixedLayerFromDataset->dataset()->ijToInlineXlineTransfoForInline();
		}
		else if (pRGBLayerImplFreeHorizonOnSlice)
		{
			transform = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->ijToInlineXlineTransfoForInline();
		}
		else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
		{
			transform = pFixedLayersImplFreeHorizonFromDatasetAndCube->ijToInlineXlineTransfoForInline();
		}
		return transform;
}



const Affine2DTransformation* FreeHorizon::Attribut::ijToInlineXlineTransfoForXline() const
{
	const Affine2DTransformation* transform = nullptr;
		if (pFixedRGBLayersFromDatasetAndCube)
		{
			transform = pFixedRGBLayersFromDatasetAndCube->ijToInlineXlineTransfoForXline();
		}
		else if (pFixedLayerFromDataset)
		{
			transform = pFixedLayerFromDataset->dataset()->ijToInlineXlineTransfoForXline();
		}
		else if (pRGBLayerImplFreeHorizonOnSlice)
		{
			transform = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->ijToInlineXlineTransfoForXline();
		}
		else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
		{
			transform = pFixedLayersImplFreeHorizonFromDatasetAndCube->ijToInlineXlineTransfoForXline();
		}
		return transform;
}

bool FreeHorizon::Attribut::isIsoInT() const
{
	// maybe use Iso buffer and test step and origin
	bool val = true;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->isIsoInT();
	}
	else if (pFixedLayerFromDataset)
	{
		val = true;
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = false;
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->isIsoInT();
	}
	return val;
}

bool FreeHorizon::Attribut::isIndexCache(int index) const
{
	bool val = false;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->isIndexCache(index);
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->isIndexCache(index);
	}
	return val;
}

int FreeHorizon::Attribut::currentImageIndex() const
{
	int val = -1;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->currentImageIndex();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->currentImageIndex();
	}
	return val;
}

SurfaceMeshCache* FreeHorizon::Attribut::getMeshCache(int index) const
{
	SurfaceMeshCache* val = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->getMeshCache(index);
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->getMeshCache(index);
	}
	return val;
}

int FreeHorizon::Attribut::getSimplifyMeshSteps() const
{
	int val = 10;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->getSimplifyMeshSteps();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->getSimplifyMeshSteps();
	}
	return val;
}

int FreeHorizon::Attribut::getCompressionMesh()const
{
	int val = 10;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->getCompressionMesh();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->getCompressionMesh();
	}
	return val;
}

int FreeHorizon::Attribut::width() const
{
	int val = 0;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->width();
	}
	else if (pFixedLayerFromDataset)
	{
		val = pFixedLayerFromDataset->width();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->width();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->width();
	}
	return val;
}

int FreeHorizon::Attribut::depth() const
{
	int val = 0;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->depth();
	}
	else if (pFixedLayerFromDataset)
	{
		val = pFixedLayerFromDataset->depth();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->depth();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->depth();
	}
	return val;
}

int FreeHorizon::Attribut::heightFor3D() const
{
	int val = 0;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->heightFor3D();
	}
	else if (pFixedLayerFromDataset)
	{
		val = pFixedLayerFromDataset->dataset()->height();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = pRGBLayerImplFreeHorizonOnSlice->layerSlice()->seismic()->height();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->heightFor3D();
	}
	return val;
}

ImageFormats::QSampleType FreeHorizon::Attribut::imageType() const
{
	ImageFormats::QSampleType val = ImageFormats::QSampleType::ERR;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->image()->sampleType();
	}
	else if (pFixedLayerFromDataset && pFixedLayerFromDataset->keys().size()>0)
	{
		val = pFixedLayerFromDataset->image(pFixedLayerFromDataset->keys()[0])->sampleType();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = pRGBLayerImplFreeHorizonOnSlice->image()->get(0)->sampleType();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->image()->sampleType();
	}
	return val;
}

ImageFormats::QSampleType FreeHorizon::Attribut::isoType() const
{
	ImageFormats::QSampleType val = ImageFormats::QSampleType::ERR;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		val = pFixedRGBLayersFromDatasetAndCube->isoSurfaceHolder()->sampleType();
	}
	else if (pFixedLayerFromDataset)
	{
		val = pFixedLayerFromDataset->getIsoBuffer().buffer->sampleType();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		val = pRGBLayerImplFreeHorizonOnSlice->getIsoBuffer().buffer->sampleType();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		val = pFixedLayersImplFreeHorizonFromDatasetAndCube->isoSurfaceHolder()->sampleType();
	}
	return val;
}

void FreeHorizon::Attribut::copyImageData(CUDARGBInterleavedImage* data)
{
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		pFixedRGBLayersFromDatasetAndCube->initialize();
		CUDARGBInterleavedImage* image = pFixedRGBLayersFromDatasetAndCube->image();
		QVector2D rangeRed =  image->redRange();
		QVector2D rangeGreen =  image->greenRange();
		QVector2D rangeBlue =  image->blueRange();
		image->lockPointer();
		QByteArray array = image->byteArray();
		data->updateTexture(array,false,rangeRed,rangeGreen,rangeBlue);
		image->unlockPointer();
	}
	else if (pFixedLayerFromDataset && pFixedLayerFromDataset->keys().size()>0)
	{
		CUDAImagePaletteHolder* image = pFixedLayerFromDataset->image(pFixedLayerFromDataset->keys()[0]);
		QByteArray buf;
		buf.resize(image->width()*image->height()*image->sampleType().byte_size()*3);
		long N = image->width()*image->height();
		long typeByteSize = image->sampleType().byte_size();
		image->lockPointer();
		char* tab = static_cast<char*>(image->backingPointer());

		#pragma omp parallel
		for (long i=0; i<N; i++)
		{
			for (int c=0; c<3; c++)
			{
				for(long j=0; j<typeByteSize; j++)
				{
					buf[(i*3+c)*typeByteSize+j] = tab[i*typeByteSize+j];
				}
			}
		}
		image->unlockPointer();
		data->updateTexture(buf,false,image->range(),image->range(),image->range());
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		CUDARGBImage* image = pRGBLayerImplFreeHorizonOnSlice->image();
		QByteArray buf;
		buf.resize(image->width()*image->height()*image->get(0)->sampleType().byte_size()*3);
		for (int c=0; c<3; c++)
		{
			long N = image->width()*image->height();
			long typeByteSize = image->get(0)->sampleType().byte_size();
			image->get(c)->lockPointer();
			char* tab = static_cast<char*>(image->get(c)->backingPointer());

			#pragma omp parallel
			for (long i=0; i<N; i++)
			{
				for(long j=0; j<typeByteSize; j++)
				{
					buf[(i*3+c)*typeByteSize+j] = tab[i*typeByteSize+j];
				}
			}
			image->get(c)->unlockPointer();
		}
		data->updateTexture(buf,false,image->get(0)->range(),image->get(1)->range(),image->get(2)->range());
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		pFixedLayersImplFreeHorizonFromDatasetAndCube->initialize();
		CPUImagePaletteHolder* image = pFixedLayersImplFreeHorizonFromDatasetAndCube->image();
		QByteArray buf;
		buf.resize(image->width()*image->height()*image->sampleType().byte_size()*3);
		long N = image->width()*image->height();
		long typeByteSize = image->sampleType().byte_size();
		image->lockPointer();
		const char* tab = static_cast<const char*>(image->constBackingPointer());

		#pragma omp parallel
		for (long i=0; i<N; i++)
		{
			for (int c=0; c<3; c++)
			{
				for(long j=0; j<typeByteSize; j++)
				{
					buf[(i*3+c)*typeByteSize+j] = tab[i*typeByteSize+j];
				}
			}
		}
		image->unlockPointer();
		data->updateTexture(buf,false,image->range(),image->range(),image->range());

	}
}

void FreeHorizon::Attribut::copyIsoData(CPUImagePaletteHolder* data)
{
	CPUImagePaletteHolder* isoCurrent = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		pFixedRGBLayersFromDatasetAndCube->initialize();
		isoCurrent = pFixedRGBLayersFromDatasetAndCube->isoSurfaceHolder();
	}
	else if (pFixedLayerFromDataset)
	{
		isoCurrent = pFixedLayerFromDataset->getIsoBuffer().buffer.get();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		isoCurrent = pRGBLayerImplFreeHorizonOnSlice->getIsoBuffer().buffer.get();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		pFixedLayersImplFreeHorizonFromDatasetAndCube->initialize();
		isoCurrent = pFixedLayersImplFreeHorizonFromDatasetAndCube->isoSurfaceHolder();
	}
	if (isoCurrent!=nullptr)
	{
		QVector2D range =  isoCurrent->range();
		QByteArray arrayIso = isoCurrent->getDataAsByteArray();
		data->updateTexture(arrayIso, false,range);
	}
}


CPUImagePaletteHolder* FreeHorizon::Attribut::getIsoSurface()
{
	if (pFixedRGBLayersFromDatasetAndCube)
	{
		pFixedRGBLayersFromDatasetAndCube->initialize();
		return pFixedRGBLayersFromDatasetAndCube->isoSurfaceHolder();
	}
	else if (pFixedLayerFromDataset)
	{
		return pFixedLayerFromDataset->getIsoBuffer().buffer.get();
	}
	else if (pRGBLayerImplFreeHorizonOnSlice)
	{
		return pRGBLayerImplFreeHorizonOnSlice->getIsoBuffer().buffer.get();
	}
	else if (pFixedLayersImplFreeHorizonFromDatasetAndCube)
	{
		pFixedLayersImplFreeHorizonFromDatasetAndCube->initialize();
		return pFixedLayersImplFreeHorizonFromDatasetAndCube->isoSurfaceHolder();
	}
	return nullptr;
}

/*
void FreeHorizon::freeHorizonAttributCreate()
{
	std::vector<QString> list = FreeHorizonQManager::getAttributData(m_path);
	m_attribut.resize(list.size());
	for (int n=0; n<list.size(); n++)
	{
		QFileInfo fi(list[n]);
		QString name = fi.completeBaseName();
		// m_attribut[n] = new FreeHorizonAttribut(m_workingSet, m_survey, m_path, name, m_parent);
		bool isValid = false;
		QString datasetName = QString::fromStdString(FreeHorizonManager::dataSetNameWithPrefixGet(m_path.toStdString()));
		QString datasetPath0 = m_survey->idPath() + "/DATA/SEISMIC/" + datasetName + ".xt";

		if ( !QFile::exists(datasetPath0) )
		{
			datasetPath0 = findCompatibleDataSetForHorizon(m_path, m_survey->idPath() + "/DATA/SEISMIC/");
		}

		if ( QFile::exists(datasetPath0) )
		{
			QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(name.toStdString()));
			if ( type == "isochrone" || type == "mean" )
			{
				FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, m_survey, &isValid);
				std::vector<QString> names;
				names.resize(1, "a");
				m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);

				// Seismic3DDataset *dataset = new Seismic3DDataset(m_survey, name, m_workingSet, Seismic3DAbstractDataset::Seismic, datasetPath0, m_parent);
				// m_attribut[n].pFixedLayerFromDataset = new FixedLayerImplFreeHorizonFromDataset(name, m_workingSet, dataset, m_parent);
			}

			else if ( type == "gcc" )
			{
				FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, m_survey, &isValid);
				std::vector<QString> names;
				names.resize(1, "a");
				m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
			}

			else if ( type == "spectrum" )
			{
				FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, m_survey, &isValid);
				std::vector<QString> names;
				names.resize(1, "a");
				m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
			}
		}
	}
}
*/

void FreeHorizon::freeHorizonAttributCreate()
{
	QString isoPath = m_path + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);
	if (!QFile::exists(isoPath))
	{
		return;
	}
	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();


	std::vector<QString> list = FreeHorizonQManager::getAttributData(m_path);
	m_attribut.resize(list.size());
	//qDebug()<<"start freeHorizonAttributCreate"<<list.size();
	for (int n=0; n<list.size(); n++)
	{


		//std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
		QFileInfo fi(list[n]);
		QString name = fi.completeBaseName();
		// m_attribut[n] = new FreeHorizonAttribut(m_workingSet, m_survey, m_path, name, m_parent);
		bool isValid = false;

		QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(name.toStdString()));
	//	qDebug()<<"-----> create Attribut : "<<type;
		if ( type == "isochrone" || type == "mean" || type == "gcc" )
		{
			FixedLayersFromDatasetAndCube::Grid3DParameter params = FixedLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			std::vector<QString> names;
			names.resize(1, "a");
			//m_attribut[n].pFixedLayersImplFreeHorizonFromDatasetAndCube = new FixedLayerImplFreeHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent);
			m_attribut[n].setFixedLayersImplFreeHorizonFromDatasetAndCube (new FixedLayerImplFreeHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent));
			if (m_isoData.isNull() && type == "isochrone") {
				m_isoData = m_attribut[n].getFixedLayersImplFreeHorizonFromDatasetAndCube();
				QString toolTip = this->name();
				QStringList parenthesisSplit = toolTip.split("(");
				if (parenthesisSplit.size()>1) {
					toolTip = parenthesisSplit[0];
				}
				m_isoData->setSectionToolTip(toolTip);
				connect(m_isoData.data(), &FixedLayerImplFreeHorizonFromDatasetAndCube::colorChanged,
						this, &FreeHorizon::setColor);
			}
		}
/*
		else if ( type == "gcc" )
		{
			FixedLayersFromDatasetAndCube::Grid3DParameter params = FixedLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, m_survey, &isValid);
			std::vector<QString> names;
			// m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
			m_attribut[n].pFixedLayersImplFreeHorizonFromDatasetAndCube = new
					FixedLayerImplFreeHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent);
		}
		*/

		else if ( type == "spectrum" )
		{
			FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			std::vector<QString> names;
			names.resize(1, "a");
			//m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
			m_attribut[n].setFixedRGBLayersFromDatasetAndCube (new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent));
		}
	//	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	//		qDebug() <<"type:"<<type<<  "finish  : " << std::chrono::duration<double, std::milli>(end2-start2).count();

	}
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//	qDebug() << "finish freeHorizonAttributCreate : " << std::chrono::duration<double, std::milli>(end-start).count();
}

// !! name + ext
bool FreeHorizon::isAttributExists(QString name)
{
	for (int i=0; i<m_attribut.size(); i++)
	{
		QFileInfo fi(name);
		QString name0 = fi.completeBaseName();
		if ( m_attribut[i].name() == name0 )
			return true;
	}
	return false;
}


void FreeHorizon::freeHorizonAttributCreate(QString filename)
{
	qDebug()<<"start freeHorizonAttributCreate 2   ==>"<<filename;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	QFileInfo fi(filename);
	QString name = fi.completeBaseName();
	// m_attribut[n] = new FreeHorizonAttribut(m_workingSet, m_survey, m_path, name, m_parent);
	bool isValid = false;
	QString isoPath = m_path + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);

	if ( QFile::exists(isoPath) )
	{
		QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(name.toStdString()));
		if ( type == "isochrone" || type == "mean" || type == "gcc" )
		{
			FixedLayersFromDatasetAndCube::Grid3DParameter params = FixedLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			std::vector<QString> names;
			names.resize(1, "a");
			Attribut att;
			att.setFixedLayersImplFreeHorizonFromDatasetAndCube (new FixedLayerImplFreeHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent));
			if (m_isoData.isNull() && type == "isochrone") {
				m_isoData = att.getFixedLayersImplFreeHorizonFromDatasetAndCube();
				connect(m_isoData.data(), &FixedLayerImplFreeHorizonFromDatasetAndCube::colorChanged,
						this, &FreeHorizon::setColor);
			}
			m_attribut.push_back(att);
			emit attributAdded(&m_attribut[m_attribut.size()-1]);
			// emit freeHorizonAttributAdded(&att);
		}
		/*
			else if ( type == "gcc" )
			{
				FixedLayersFromDatasetAndCube::Grid3DParameter params = FixedLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, m_survey, &isValid);
				std::vector<QString> names;
				// m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
				m_attribut[n].pFixedLayersImplFreeHorizonFromDatasetAndCube = new
						FixedLayerImplFreeHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent);
			}
		 */

		else if ( type == "spectrum" )
		{
			FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			std::vector<QString> names;
			names.resize(1, "a");
			Attribut att;
			//m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent);
			att.setFixedRGBLayersFromDatasetAndCube (new FixedAttributImplFreeHorizonFromDirectories(m_path, name, names, m_workingSet, params, m_parent));
			m_attribut.push_back(att);
			emit attributAdded(&m_attribut[m_attribut.size()-1]);
		}
	}
}


int FreeHorizon::attributListRemove(QString name)
{
	int idx = -1;
	for (int i=0; i<m_attribut.size(); i++)
	{
		if ( m_attribut[i].name() == name )
		{
			idx = i;
			break;
		}
	}
	if ( idx < 0 ) return 0;
	m_attribut.erase(m_attribut.begin()+idx, m_attribut.begin()+idx+1);
	return 1;
}


void FreeHorizon::freeHorizonAttributRemove(QString path)
{
	QFileInfo fi(path);
	QString name = fi.completeBaseName();

	for (int i=0; i<m_attribut.size(); i++)
	{
		if ( m_attribut[i].name() == name )
		{
			Attribut *p = &m_attribut[i];
			emit attributRemoved(p);
			attributListRemove(name);
		}
	}
}



	//IData
 IGraphicRepFactory *FreeHorizon::graphicRepFactory() {
	return m_repFactory;
}


QUuid FreeHorizon::dataID() const {
	return m_uuid;
}

QColor FreeHorizon::color() const {
	return m_color;
}

void FreeHorizon::setColor(const QColor& color) {
	if (m_color!=color) {
		m_color = color;
		emit colorChanged(m_color);
	}
}

ITreeWidgetItemDecorator* FreeHorizon::getTreeWidgetItemDecorator()
{
	if (m_decorator==nullptr)
	{
		QIcon icon = FreeHorizonQManager::getHorizonIcon(m_color, m_sampleUnit);
		m_decorator = new IconTreeWidgetItemDecorator(icon, this);
		connect(this, &FreeHorizon::iconChanged, m_decorator, &IconTreeWidgetItemDecorator::setIcon);
		connect(this, &FreeHorizon::colorChanged, this, &FreeHorizon::updateIcon);
	}
	return m_decorator;
}

IData* FreeHorizon::getIsochronData() {
	return m_isoData.data();
}

QString FreeHorizon::findCompatibleDataSetForHorizon(QString horizonPath, QString dataSetPath)
{
	QDir dir(dataSetPath);
	QFileInfoList xtInfoList = dir.entryInfoList(QStringList() << "*.xt", QDir::Files | QDir::Readable);
	for (int i=0; i<xtInfoList.size(); i++)
	{
		QString filename = xtInfoList[i].filePath();
		if ( FreeHorizonManager::dataSetvsHorizonCompatibility(filename.toStdString(), horizonPath.toStdString()) ) return filename;

	}
	return "";
}

void FreeHorizon::updateIcon(QColor color)
{
	QIcon icon = FreeHorizonQManager::getHorizonIcon(color, m_sampleUnit);
	emit iconChanged(icon);
}

FreeHorizon::Attribut FreeHorizon::attribut(int index) const
{
	Attribut attr;
	if (index>=0 && index<m_attribut.size()) {
		attr = m_attribut[index];
	}
	return attr;
}

long FreeHorizon::numAttributs() const
{
	return m_attribut.size();
}

void FreeHorizon::openSismageExporter(QWidget* parent)
{
	ExportNVHorizonDialog* dialog = new ExportNVHorizonDialog(this);
	dialog->setAttribute(Qt::WA_DeleteOnClose);
	dialog->show();
}

SeismicSurvey* FreeHorizon::survey() const
{
	return m_survey;
}

/*
QList<RgtSeed> Marker::getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit) {
	QList<RgtSeed> outList;
	for (WellPick* pick : m_wellPicks) {
		std::pair<RgtSeed, bool> projection = pick->getProjectionOnDataset(dataset, channel, sampleUnit);
		if (projection.second) {
			outList.push_back(projection.first);
		}
	}
	return outList;
}

QList<WellPick*> Marker::getWellPickFromWell(WellBore* bore) {
	QList<WellPick*> out;
	for (std::size_t i=0; i<m_wellPicks.size(); i++) {
		if (m_wellPicks[i]->wellBore()==bore) {
			out.push_back(m_wellPicks[i]);
		}
	}
	return out;
}

const QList<WellPick*>& Marker::wellPicks() const {
	return m_wellPicks;
}

void Marker::addWellPick(WellPick* pick) {
	m_wellPicks.push_back(pick);
	emit wellPickAdded(pick);
}

void Marker::removeWellPick(WellPick* pick) {
	m_wellPicks.removeOne(pick);
	emit wellPickRemoved(pick);
}
*/

//void FreeHorizon::freeHorizonAdded()
//{
////		qDebug()<<"SeismicSurvey::addDataset "<<dataset->name();
////		m_datasets.push_back(dataset);
////		emit datasetAdded(dataset);
//}
