
#include <Xt.h>
#include "seismicsurvey.h"
#include "smdataset3D.h"
#include "gdal.h"
#include <igraphicrepfactory.h>
#include "workingsetmanager.h"
#include "GeotimeProjectManagerWidget.h"
#include <freeHorizonManager.h>
#include <freehorizonattributlayer.h>


FreeHorizonAttributLayer::FreeHorizonAttributLayer(QString path, QString name,
		WorkingSetManager *workingSet, SeismicSurvey *survey, FixedRGBLayersFromDatasetAndCube::Grid3DParameter param, QObject *parent) :
		FixedRGBLayersFromDatasetAndCube(name, workingSet, param, parent) //IData(workingSet, parent)
{
	m_path = path;
	m_survey = survey;
	m_name = name;
	bool ok = false;
	m_numLayers = 1;
	m_isoOrigin = 0;
	initLayersList();
	setCurrentImageIndex(0);

	// FixedRGBLayersFromDatasetAndCube::Grid3DParameter *params = new FixedRGBLayersFromDatasetAndCube::Grid3DParameter()
	// Grid3DParameter param = createGrid3DParameter(path, survey, &ok);
}

FreeHorizonAttributLayer::~FreeHorizonAttributLayer()
{

}




/*
FreeHorizonAttributLayer::Grid3DParameter FreeHorizonAttributLayer::createGrid3DParameter( const QString& datasetPath, SeismicSurvey* survey, bool* ok)
{
	Grid3DParameter params;
	*ok = true;

	QString datasetName = QString::fromStdString(FreeHorizonManager::dataSetNameGet(m_path.toStdString()));
	QString datasetPath0 = survey->idPath() + "/DATA/SEISMIC/" + datasetName + ".xt";

	inri::Xt xt(datasetPath0.toStdString().c_str());
	if (xt.is_valid()) {// check if xt valid before using SmDataset3D because it use xt class but does not check validity
		params.cubeSeismicAddon.set(
				xt.startSamples(), xt.stepSamples(),
				xt.startRecord(), xt.stepRecords(),
				xt.startSlice(), xt.stepSlices());
		params.width = xt.nRecords();
		params.depth = xt.nSlices();
		params.heightFor3D = xt.nSamples();
		int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(datasetPath0);
		params.cubeSeismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);
		*ok = timeOrDepth==0 || timeOrDepth==1;
	} else {
		std::cerr << "xt cube is not valid (" << datasetPath0.toStdString() << ")" << std::endl;
		*ok = false;
	}

	if (*ok) {
		// get transforms
		SmDataset3D d3d(datasetPath0.toStdString());
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
*/


IGraphicRepFactory* FreeHorizonAttributLayer::graphicRepFactory() {
	return nullptr; //m_repFactory.get();
}


void FreeHorizonAttributLayer::setCurrentImageIndexInternal(long newIndex)
{

}

QUuid FreeHorizonAttributLayer::dataID() const
{
	return m_uuid;
}

QString FreeHorizonAttributLayer::name() const
{
	return m_name;
}


void FreeHorizonAttributLayer::nextCurrentIndex()
{

}

std::vector<StackType> FreeHorizonAttributLayer::stackTypes() const
{
	std::vector<StackType> ret;
	return ret;
}

QString FreeHorizonAttributLayer::getObjFile(int index) const
{
	return "";
}

std::shared_ptr<AbstractStack> FreeHorizonAttributLayer::stack(StackType type)
{
	std::shared_ptr<AbstractStack> ret;

	return ret;
}


IsoSurfaceBuffer FreeHorizonAttributLayer::getIsoBuffer()
{
	IsoSurfaceBuffer ret;
	return ret;
}


void FreeHorizonAttributLayer::setCompressionMesh( int compress)
{

}

void FreeHorizonAttributLayer::setSimplifyMeshSteps(int steps)
{

}

QString FreeHorizonAttributLayer::surveyPath() const
{
	return m_survey->idPath();
}






// =======================================================
QString FreeHorizonAttributLayer::readAttributFromFile(int index, void *buff, long size)
{
	/*
	int idx =  getOptionAttribut();
	QString filename = getAttributFileFromIndex(index);
	if ( idx == 0 )
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
	else if ( idx == 1 || idx == 2 )
	{
		FILE* pf = fopen(filename.toStdString().c_str(), "r");
		if ( pf != nullptr )
		{
			fread(buff, sizeof(short), size*3, pf);
			fclose(pf);
			return "ok";
		}
		else
		{
			memset(buff, 0, size*3*sizeof(short));
			return "error";
		}
	}
	else if ( idx == 3 )
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
	 * */
	return "ok";
}

void FreeHorizonAttributLayer::getImageForIndex(long newIndex,
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

bool FreeHorizonAttributLayer::getImageForIndex(long newIndex,
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
