#ifndef VideoLayerRep_H
#define VideoLayerRep_H

#include <QObject>
#include <QUuid>
#include "abstractgraphicrep.h"
#include "videolayer.h"

class VideoLayerLayer;
class VideoLayerPropPanel;

class QMediaPlayer;


class VideoLayerRep: public AbstractGraphicRep {
Q_OBJECT
public:
	VideoLayerRep(VideoLayer *data, AbstractInnerView *parent = 0);
	virtual ~VideoLayerRep();


	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	IData* data() const override;
	QMediaPlayer* mediaPlayer();

	VideoLayer* videoLayer();
	virtual TypeRep getTypeGraphicRep() override;
private:
	void refreshLayer();

private:
	VideoLayerPropPanel *m_propPanel;
	VideoLayerLayer *m_layer;
	VideoLayer *m_data;

	QMediaPlayer* m_player;
};

#endif
