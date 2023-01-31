#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include <cmath>
#include <iostream>
#include "gdal.h"

#include "seismicsurveygrahicrepfactory.h"

SeismicSurvey::SeismicSurvey(WorkingSetManager *workingSet, const QString &name,
		int width, int height, QString idPath, QObject *parent) :
		IData(workingSet, parent), IFileBasedData(idPath) {
	m_name = name;
	m_width = width;
	m_height = height;
	m_uuid = QUuid::createUuid();

	m_repFactory = new SeismicSurveyGraphicRepFactory(this);

	m_ilXlToXY = new Affine2DTransformation(width, height, this);
	m_ijToXY = new Affine2DTransformation(width, height, this);
}

const Affine2DTransformation* const SeismicSurvey::inlineXlineToXYTransfo() const {
	return m_ilXlToXY;
}
const Affine2DTransformation* const SeismicSurvey::ijToXYTransfo() const {
	return m_ijToXY;
}

void SeismicSurvey::setInlineXlineToXYTransfo(
		const Affine2DTransformation &transfo) {
	m_ilXlToXY->deleteLater();
	m_ilXlToXY = new Affine2DTransformation(transfo);
	m_ilXlToXY->setParent(this);
}
void SeismicSurvey::setIJToXYTransfo(
		const Affine2DTransformation &transfo) {
	m_ijToXY->deleteLater();
	m_ijToXY = new Affine2DTransformation(transfo);
	m_ijToXY->setParent(this);
}

IGraphicRepFactory* SeismicSurvey::graphicRepFactory() {
	return m_repFactory;
}

QUuid SeismicSurvey::dataID() const {
	return m_uuid;
}

SeismicSurvey::~SeismicSurvey() {

}

void SeismicSurvey::addDataset(Seismic3DAbstractDataset *dataset) {
	qDebug()<<"SeismicSurvey::addDataset "<<dataset->name();
	m_datasets.push_back(dataset);
	emit datasetAdded(dataset);
}

void SeismicSurvey::removeDataset(Seismic3DAbstractDataset *dataset) {
	m_datasets.removeOne(dataset);
	emit datasetRemoved(dataset);
}

QList<Seismic3DAbstractDataset*> SeismicSurvey::datasets() {
	return m_datasets;
}

