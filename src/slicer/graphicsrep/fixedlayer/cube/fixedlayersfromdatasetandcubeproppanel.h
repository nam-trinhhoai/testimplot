#ifndef FixedLayersFromDatasetAndCubePropPanel_H
#define FixedLayersFromDatasetAndCubePropPanel_H

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

class FixedLayersFromDatasetAndCubePropPanel : public QWidget{
	  Q_OBJECT
public:
	  FixedLayersFromDatasetAndCubePropPanel(FixedLayersFromDatasetAndCubeRep *rep,QWidget *parent, bool enable = true);
	virtual ~FixedLayersFromDatasetAndCubePropPanel();

	virtual void updatePalette();

private:
	virtual void changeDataKeyFromSlider(long index);
	virtual void changeDataKeyFromSliderInternal(long index);
	virtual void changeDataKeyFromSpinBox();
	virtual void multiplierChanged(int index);
	virtual void modeChangedInternal(int index);
	virtual void modeChanged();

	virtual void initProgressBar(int min, int max, int val);
	virtual void valueProgressBarChanged(int val);
	virtual void endProgressBar();

	virtual void updateLockCheckBox();
	virtual void lockPalette(int state);
	virtual void updateLockRange(const QVector2D &);
	virtual void updateLockLookupTable(const LookupTable& lookupTable);

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
};

#endif
