#include "videolayerlayer.h"
#include "videolayerrep.h"
#include "videolayer.h"
#include "affine2dtransformation.h"

#include <QGraphicsVideoItem>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QMediaPlayer>


VideoLayerLayer::VideoLayerLayer(VideoLayerRep *rep,QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) : GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new QGraphicsVideoItem(parent);
	m_item->setZValue(defaultZDepth);
	m_rep->mediaPlayer()->setVideoOutput(m_item);
	m_item->setSize(QSize(m_rep->videoLayer()->width(), m_rep->videoLayer()->height()));

	const Affine2DTransformation* affine2D = m_rep->videoLayer()->ijToXYTransfo();

	QTransform transform(affine2D->imageToWorldTransformation().toTransform());
	m_item->setTransform(transform);
}

VideoLayerLayer::~VideoLayerLayer() {}

void VideoLayerLayer::show() {
	m_scene->addItem(m_item);
	m_item->show();
}

void VideoLayerLayer::hide() {
	m_scene->removeItem(m_item);
}

QRectF VideoLayerLayer::boundingRect() const {
	return m_item->boundingRect();
}

void VideoLayerLayer::refresh() {
	m_item->update();
}
