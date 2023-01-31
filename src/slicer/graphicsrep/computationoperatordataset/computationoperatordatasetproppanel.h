#ifndef ComputationOperatorDatasetPropPanel_H
#define ComputationOperatorDatasetPropPanel_H

#include <QWidget>

class ComputationOperatorDatasetRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;

class ComputationOperatorDatasetPropPanel: public QWidget {
	Q_OBJECT
public:
	ComputationOperatorDatasetPropPanel(ComputationOperatorDatasetRep *rep, QWidget *parent = 0);
	void updatePalette();

private slots:
	void showColorScale(int value);
	void updateChannel(int channel);
	void updateChannelInternal(int channel);

	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);

private:
	ComputationOperatorDatasetRep *m_rep;
	PaletteWidget *m_palette;
	QCheckBox * m_showColorScale;
	QCheckBox* m_lockPalette;
	QSlider* m_channelSlider;
	QSpinBox* m_channelSpinBox;
};

#endif
