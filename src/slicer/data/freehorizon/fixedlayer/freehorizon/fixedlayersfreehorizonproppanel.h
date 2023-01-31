
#ifndef FixedLayersFreeHorizonPropPanel_H
#define FixedLayersFreeHorizonPropPanel_H

#include <QWidget>
#include <QMutex>

#include "lookuptable.h"

class FixedLayersFromDatasetAndCubeRep;
class QSlider;
class QSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;
class QComboBox;
class QCheckBox;
class EditingSpinBox;
class QProgressBar;
class PaletteWidget;

#include <fixedlayersfromdatasetandcubeproppanel.h>

class FixedLayersFreeHorizonPropPanel : public FixedLayersFromDatasetAndCubePropPanel {
	  Q_OBJECT
public:
	  FixedLayersFreeHorizonPropPanel(FixedLayersFromDatasetAndCubeRep *rep,QWidget *parent);
	virtual ~FixedLayersFreeHorizonPropPanel();

	void updatePalette();

private:
	void changeDataKeyFromSlider(long index);
	void changeDataKeyFromSliderInternal(long index);
	void changeDataKeyFromSpinBox();
	void multiplierChanged(int index);
	void modeChangedInternal(int index);
	void modeChanged();

	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int val);
	void endProgressBar();

	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);
	void updateLockLookupTable(const LookupTable& lookupTable);

	FixedLayersFromDatasetAndCubeRep *m_rep;
	PaletteWidget *m_palette;
	EditingSpinBox* m_layerNameSpinBox;
	QSlider* m_slider;
	QComboBox* m_multiplierComboBox;
	QComboBox* m_modeComboBox;
	QProgressBar* m_progressBar;
	QCheckBox* m_lockPalette;

	QToolButton* m_playButton;
	QToolButton* m_loopButton;

	QSpinBox* m_simplifyStepsSpinBox;

	int m_stepMultiplier = 1;
	QMutex m_mutex;
	QMutex m_mutexIndexList;
	QList<long> m_sliderIndexList;

	QSlider *m_scaleSlider = nullptr;
	QSpinBox* m_scaleSpinBox = nullptr;
	int m_nbreScales = -1;

private slots:
	void scaleChange(int idx);
	void scaleSpinChange(int idx);

};

#endif
