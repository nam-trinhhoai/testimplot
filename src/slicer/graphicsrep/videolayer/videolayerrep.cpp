#include "videolayerrep.h"

#include "videolayerlayer.h"
#include "videolayerproppanel.h"
#include "abstractinnerview.h"

#include <QMediaPlayer>
#include <QAudioOutput>
#include <QGraphicsScene>

VideoLayerRep::VideoLayerRep(VideoLayer *data, AbstractInnerView *parent) : AbstractGraphicRep(parent) {
	m_data = data;
	m_name = m_data->name();

	m_propPanel = nullptr;
	m_layer = nullptr;

	m_player = new QMediaPlayer(this);
	//m_player->setMedia(QUrl(m_data->mediaPath()));
	m_player->setSource(QUrl(m_data->mediaPath()));
	//m_player->setMuted(true);
	m_player->audioOutput()->setMuted(true);
}

VideoLayerRep::~VideoLayerRep() {
	if (m_layer) {
		delete m_layer;
	}
	if (m_propPanel) {
		m_propPanel->deleteLater();
	}
}

//AbstractGraphicRep
QWidget* VideoLayerRep::propertyPanel() {
	if (m_propPanel==nullptr) {
		m_propPanel = new VideoLayerPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}

GraphicLayer* VideoLayerRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer==nullptr) {
		m_layer = new VideoLayerLayer(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

IData* VideoLayerRep::data() const {
	return m_data;
}

QMediaPlayer* VideoLayerRep::mediaPlayer() {
	return m_player;
}

VideoLayer* VideoLayerRep::videoLayer() {
	return m_data;
}

void VideoLayerRep::refreshLayer() {
	if (m_layer) {
		m_layer->refresh();
	}
}

AbstractGraphicRep::TypeRep VideoLayerRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Video;
}
