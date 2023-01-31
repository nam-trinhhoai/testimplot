#ifndef HorizonFolderPropPanelOnSlice_H
#define HorizonFolderPropPanelOnSlice_H

#include <QMutex>
#include <QWidget>

#include "lookuptable.h"

class QLineEdit;
//class HorizonFolderDataRep;
class HorizonFolderRepOnSlice;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class QListWidget;
class QPushButton;
class QToolButton;
class QTimer;
class QComboBox;
class RGBPaletteWidget;
class PaletteWidget;
class CUDARGBInterleavedImage;
class IData;

class HorizonFolderPropPanelOnSlice: public QWidget {
	Q_OBJECT
public:
	HorizonFolderPropPanelOnSlice(HorizonFolderRepOnSlice *rep,QWidget *parent);
	virtual ~HorizonFolderPropPanelOnSlice();

	//void updatePalette(CUDARGBInterleavedImage*);

private slots:

//	void showHorizonWidget();

	void add();
	void remove();

	void moveUp();
	void moveDown();
	void moveTop();
	void moveBottom();

	//	void addView3D();

	QStringList getAttributesAvailable();

//	void filterChanged();
	void trt_basketListSelectionChanged();
	void playAnimation(bool );
	void moveAnimation(int);
	void updateAnimation();

	void speedChanged(int);

	void dataRemoved(IData* );
	void dataAdded(IData*);
	void orderChangedFromData(int oldIndex, int newIndex);

	//void view3DChanged(int i);
	void attributChanged(int i);


//	void setRangeFromImage(unsigned int index,QVector2D range);
//	void setRangeToImage(QVector2D range);



protected:

private:

	HorizonFolderRepOnSlice *m_rep;

	//RGBPaletteWidget* m_paletteRGB;
//	PaletteWidget* m_palette;
	CUDARGBInterleavedImage* m_lastimage = nullptr;

	//bool m_modeRGB = true;
	//bool m_lastRGB = true;

	QListWidget* m_orderListWidget;

	QComboBox* m_comboAttribut;
	//QComboBox* m_comboView3D;
	QToolButton* m_playButton;
	QSlider* m_animSlider;
	QSlider* m_speedSlider;

	QTimer* m_animTimer;
	int m_speedAnim = 500;
	int m_lastSelected = -1;

	int m_lastValue=-1;

	QMutex m_orderMutex;
};

#endif

/*#include <QWidget>

class HorizonFolderRepOnSlice;
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

class HorizonFolderPropPanelOnSlice : public QWidget{
	  Q_OBJECT
public:
	  HorizonFolderPropPanelOnSlice(HorizonFolderRepOnSlice *rep,QWidget *parent);
	virtual ~HorizonFolderPropPanelOnSlice();

private:
	void changeDataKeyFromSlider(long index);
	void changeDataKeyFromSpinBox();
	void multiplierChanged(int index);
	void modeChangedInternal(int index);
	void modeChanged();

	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int val);
	void endProgressBar();

	HorizonFolderRepOnSlice *m_rep;
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
*/
