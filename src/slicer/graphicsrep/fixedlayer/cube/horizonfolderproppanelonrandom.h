#ifndef HorizonFolderPropPanelOnRandom_H
#define HorizonFolderPropPanelOnRandom_H

#include <QMutex>
#include <QWidget>

#include "lookuptable.h"

class QLineEdit;
//class HorizonFolderDataRep;
class HorizonFolderRepOnRandom;
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

class HorizonFolderPropPanelOnRandom: public QWidget {
	Q_OBJECT
public:
	HorizonFolderPropPanelOnRandom(HorizonFolderRepOnRandom *rep,QWidget *parent);
	virtual ~HorizonFolderPropPanelOnRandom();

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

	HorizonFolderRepOnRandom *m_rep;

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


