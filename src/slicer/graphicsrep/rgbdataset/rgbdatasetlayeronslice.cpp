#include "rgbdatasetlayeronslice.h"
#include "rgbdatasetreponslice.h"
#include "qglfullcudargbaimageitem.h"
#include "cudaimagepaletteholder.h"
#include <QGraphicsScene>

RgbDatasetLayerOnSlice::RgbDatasetLayerOnSlice(RgbDatasetRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_parent = parent;
	m_item = new QGLFullCUDARgbaImageItem(rep->red(), rep->green(), rep->blue(), rep->alpha(), m_rep->rgbDataset()->alphaMode(), parent);
	m_item->setZValue(defaultZDepth);
	connect(rep->red(), SIGNAL(rangeChanged(const QVector2D &)), this,
						SLOT(refresh()));
	connect(rep->red(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
							SLOT(refresh()));
	connect(rep->green(), SIGNAL(rangeChanged(const QVector2D &)), this,
						SLOT(refresh()));
	connect(rep->green(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
							SLOT(refresh()));
	connect(rep->blue(), SIGNAL(rangeChanged(const QVector2D &)), this,
						SLOT(refresh()));
	connect(rep->blue(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
							SLOT(refresh()));
	if (rep->alpha()) {
		connect(rep->alpha(), SIGNAL(rangeChanged(const QVector2D &)), this,
						SLOT(refresh()));
		connect(rep->alpha(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
							SLOT(refresh()));
	} else {

	}

	m_item->setOpacity(m_rep->rgbDataset()->constantAlpha());
	m_item->setRadiusAlpha(m_rep->rgbDataset()->radiusAlpha());

	connect(m_rep->rgbDataset(), &RgbDataset::constantAlphaChanged, this, &RgbDatasetLayerOnSlice::constantAlphaChanged);
	connect(m_rep->rgbDataset(), &RgbDataset::radiusAlphaChanged, this, &RgbDatasetLayerOnSlice::radiusAlphaChanged);
	connect(m_rep->rgbDataset(), &RgbDataset::alphaModeChanged, this, &RgbDatasetLayerOnSlice::modeChanged);
}

RgbDatasetLayerOnSlice::~RgbDatasetLayerOnSlice() {
}
void RgbDatasetLayerOnSlice::show()
{
	m_scene->addItem(m_item);
	m_isShown = true;
}
void RgbDatasetLayerOnSlice::hide()
{
	m_scene->removeItem(m_item);
	m_isShown = false;
}

QRectF RgbDatasetLayerOnSlice::boundingRect() const
{
	return m_rep->red()->worldExtent();
}

void RgbDatasetLayerOnSlice::refresh() {
	m_item->update();
}

void RgbDatasetLayerOnSlice::modeChanged() {
	bool oriIsShown = m_isShown;
	if (oriIsShown) {
		hide();
	}
	QGLFullCUDARgbaImageItem* oldItem = m_item;
	m_item = new QGLFullCUDARgbaImageItem(m_rep->red(), m_rep->green(), m_rep->blue(), m_rep->alpha(), m_rep->rgbDataset()->alphaMode(), m_parent);
	m_item->setZValue(this->m_defaultZDepth);

	m_item->setOpacity(m_rep->rgbDataset()->constantAlpha());
	m_item->setRadiusAlpha(m_rep->rgbDataset()->radiusAlpha());

	oldItem->deleteLater();
	if (oriIsShown) {
		show();
	}
}

void RgbDatasetLayerOnSlice::constantAlphaChanged() {
	m_item->setOpacity(m_rep->rgbDataset()->constantAlpha());
}

void RgbDatasetLayerOnSlice::radiusAlphaChanged() {
	m_item->setRadiusAlpha(m_rep->rgbDataset()->radiusAlpha());
}

