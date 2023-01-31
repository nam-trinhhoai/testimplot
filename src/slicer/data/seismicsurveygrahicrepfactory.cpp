#include "seismicsurveygrahicrepfactory.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismicsurveyrep.h"
#include "seismicsurveyreponmap.h"
#include "seismicsurveyrepon3D.h"

SeismicSurveyGraphicRepFactory::SeismicSurveyGraphicRepFactory(
		SeismicSurvey *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	connect(m_data, SIGNAL(datasetAdded(Seismic3DAbstractDataset *)), this,
			SLOT(datasetAdded(Seismic3DAbstractDataset *)));
}

SeismicSurveyGraphicRepFactory::~SeismicSurveyGraphicRepFactory() {

}
AbstractGraphicRep* SeismicSurveyGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView)
		return new SeismicSurveyRepOnMap(m_data, parent);
	else if (type == ViewType::InlineView || type == ViewType::XLineView ||
			type == ViewType::RandomView) {
		return new SeismicSurveyRep(m_data, parent);
	} else if (type == ViewType::View3D) {
		return new SeismicSurveyRepOn3D(m_data, parent);
	}

	return nullptr;
}

QList<IGraphicRepFactory*> SeismicSurveyGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {


	QList<IGraphicRepFactory*> reps;
	for (Seismic3DAbstractDataset *d : m_data->datasets()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void SeismicSurveyGraphicRepFactory::datasetAdded(
		Seismic3DAbstractDataset *dataset) {
	emit childAdded(dataset->graphicRepFactory());
}

void SeismicSurveyGraphicRepFactory::datasetRemoved(
		Seismic3DAbstractDataset *dataset) {
	emit childRemoved(dataset->graphicRepFactory());
}
