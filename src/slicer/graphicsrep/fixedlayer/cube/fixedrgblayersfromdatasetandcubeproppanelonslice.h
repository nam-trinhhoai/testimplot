#ifndef FixedRGBLayersFromDatasetAndCubePropPanelOnSlice_H
#define FixedRGBLayersFromDatasetAndCubePropPanelOnSlice_H

#include <QWidget>

class FixedRGBLayersFromDatasetAndCubeRepOnSlice;
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

class FixedRGBLayersFromDatasetAndCubePropPanelOnSlice : public QWidget{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetAndCubePropPanelOnSlice(FixedRGBLayersFromDatasetAndCubeRepOnSlice *rep,QWidget *parent);
	virtual ~FixedRGBLayersFromDatasetAndCubePropPanelOnSlice();

private:
	void changeDataKeyFromSlider(long index);
	void changeDataKeyFromSpinBox();
	void multiplierChanged(int index);
	void modeChangedInternal(int index);
	void modeChanged();

	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int val);
	void endProgressBar();

	FixedRGBLayersFromDatasetAndCubeRepOnSlice *m_rep;
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
