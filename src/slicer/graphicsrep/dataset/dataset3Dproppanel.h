#ifndef Dataset3DPropPanel_H
#define Dataset3DPropPanel_H

#include <QWidget>

class Dataset3DSliceRep;
class PaletteWidget;
class QSpinBox;
class QSlider;

class Dataset3DPropPanel: public QWidget {
	Q_OBJECT
public:
	Dataset3DPropPanel(Dataset3DSliceRep *rep, QWidget *parent = 0);
	virtual ~Dataset3DPropPanel();
	void updatePalette();
private slots:
	void sliceChanged(int val);
	void onSliceChangedRequestFromRep(int val);
	void updateChannelInternal(int channel);
	void updateChannel(int channel);
private:
	QWidget* createSliceBox(const QString &title);
	void defineSliceMinMax(const QVector2D &imageMinMax, int step);
	void defineSliceVal(int image);
private:
	Dataset3DSliceRep *m_rep;
	PaletteWidget *m_palette;

	QSpinBox* m_sliceSpin;
	QSlider* m_sliceImageSlider;
	QSlider* m_channelSlider;
	QSpinBox* m_channelSpinBox;

};

#endif
