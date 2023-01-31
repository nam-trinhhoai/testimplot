#ifndef SlicePropPanel_H
#define SlicePropPanel_H

#include <QWidget>

class SliceRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;

class SlicePropPanel: public QWidget {
	Q_OBJECT
public:
	SlicePropPanel(SliceRep *rep, QWidget *parent = 0);
	void updatePalette();

private slots:
	void showColorScale(int value);
	void updateChannel(int channel);
	void updateChannelInternal(int channel);

	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);

	void writeRangeToFile();

private:
	SliceRep *m_rep;
	PaletteWidget *m_palette;
	QCheckBox * m_showColorScale;
	QCheckBox* m_lockPalette;
	QSlider* m_channelSlider;
	QSpinBox* m_channelSpinBox;
};

#endif
