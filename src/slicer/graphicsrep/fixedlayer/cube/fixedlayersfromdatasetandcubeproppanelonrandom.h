#ifndef FixedLayersFromDatasetAndCubePropPanelOnRandom_H
#define FixedLayersFromDatasetAndCubePropPanelOnRandom_H

#include <QWidget>

class FixedLayersFromDatasetAndCubeRepOnRandom;
class QSlider;
class QSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;
class QLabel;
class QComboBox;
class EditingSpinBox;
class QProgressBar;

class FixedLayersFromDatasetAndCubePropPanelOnRandom : public QWidget{
	  Q_OBJECT
public:
	  FixedLayersFromDatasetAndCubePropPanelOnRandom(FixedLayersFromDatasetAndCubeRepOnRandom *rep,QWidget *parent);
	virtual ~FixedLayersFromDatasetAndCubePropPanelOnRandom();

private:
	void changeDataKeyFromSlider(long index);
	void changeDataKeyFromSpinBox();
	void multiplierChanged(int index);
	void modeChangedInternal(int index);
	void modeChanged();

	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int val);
	void endProgressBar();

	FixedLayersFromDatasetAndCubeRepOnRandom *m_rep;
	EditingSpinBox* m_layerNameSpinBox;
	QSlider* m_slider;
	QComboBox* m_multiplierComboBox;
	QComboBox* m_modeComboBox;
	QProgressBar* m_progressBar;

	QToolButton* m_playButton;
	QToolButton* m_loopButton;

	int m_stepMultiplier = 1;
};

#endif
