#ifndef VideoLayerPropPanel_H
#define VideoLayerPropPanel_H

#include <QWidget>

class VideoLayerRep;
class QProgressBar;
class QComboBox;
class QSlider;
class QLineEdit;


class VideoLayerPropPanel : public QWidget {
	Q_OBJECT
public:
	VideoLayerPropPanel(VideoLayerRep* rep, QWidget* parent=0);
	virtual ~VideoLayerPropPanel();

private slots:
	void changeSpeed(int index);
	void positionChanged(qint64 position);
	void durationChanged(qint64 position);

private:
	void buildWidget();

	VideoLayerRep* m_rep;
	QComboBox* m_speedComboBox;
	QSlider* m_valueSlider;
	QLineEdit* m_valueLineEdit;
};

#endif
