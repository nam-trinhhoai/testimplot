#include "seismic3ddatasetgrahicrepfactory.h"
#include "abstractinnerview.h"
#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "slicerep.h"
#include "slicepositioncontroler.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "datasetrep.h"
#include "randomrep.h"
#include "seismic3ddataset.h"
#include "cameraparameterscontroller.h"


Seismic3DDatasetGraphicRepFactory::Seismic3DDatasetGraphicRepFactory(
		Seismic3DAbstractDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

Seismic3DDatasetGraphicRepFactory::~Seismic3DDatasetGraphicRepFactory() {

}

AbstractGraphicRep* Seismic3DDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		//TODO: add just a rectangle!
	} else if (type == ViewType::InlineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->width(), m_data->height(),
				m_data->sampleType(),
				m_data->ijToInlineXlineTransfoForInline(), parent);
		slice->setLookupTable(m_data->defaultLookupTable());

		std::array<double, 6> transfo =
				m_data->ijToInlineXlineTransfo()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
				QVector2D(transfo[3],
						transfo[3] + transfo[5] * (m_data->depth() - 1)),
				AffineTransformation(transfo[5], transfo[3]));
		SliceRep *rep = new SliceRep(m_data, slice, rangeAndStep,
				SliceDirection::Inline, parent);
		SlicePositionControler *controler = new SlicePositionControler(rep,
				parent);
		controler->setDirection(SliceDirection::Inline);
		rep->setDataControler(controler);

		return rep;
	} else if (type == ViewType::XLineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->depth(), m_data->height(),
				m_data->sampleType(),
				m_data->ijToInlineXlineTransfoForXline(), parent);
		slice->setLookupTable(m_data->defaultLookupTable());

		std::array<double, 6> transfo =
				m_data->ijToInlineXlineTransfo()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
				QVector2D(transfo[0],
						transfo[0] + transfo[1] * (m_data->width() - 1)),
				AffineTransformation(transfo[1], transfo[0]));
		SliceRep *rep = new SliceRep(m_data, slice, rangeAndStep,
				SliceDirection::XLine, parent);
		SlicePositionControler *controler = new SlicePositionControler(rep,
				parent);
		controler->setDirection(SliceDirection::XLine);
		controler->setColor(QColor(255,170,0));
		rep->setDataControler(controler);

		return rep;
	} else if (type == ViewType::View3D) {
		DatasetRep *rep = new DatasetRep(m_data, parent);
		CameraParametersController* camParametersCtrl = new CameraParametersController(rep, dynamic_cast<ViewQt3D*> (parent),parent );
		rep->setDataControler(camParametersCtrl);
		return rep;
	} else if (type == ViewType::RandomView && dynamic_cast<Seismic3DDataset*>(m_data)!=nullptr) { // CUDA part not done, restrict to cpu side
		RandomRep *rep = new RandomRep(m_data, m_data->defaultLookupTable(), parent);
		return rep;
	}
	return nullptr;
}

