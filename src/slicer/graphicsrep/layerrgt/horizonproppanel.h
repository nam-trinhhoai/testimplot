#ifndef HorizonPropPanel_H
#define HorizonPropPanel_H

#include <QWidget>
#include <QMutex>
#include <QPointer>
#include "lookuptable.h"

class QLineEdit;
//class HorizonFolderDataRep;
class HorizonDataRep;
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



class HorizonPropPanel: public QWidget {
	Q_OBJECT
public:
	HorizonPropPanel(HorizonDataRep *rep,QWidget *parent);
	virtual ~HorizonPropPanel();

	void updatePalette(CUDARGBInterleavedImage*);

	void writeAnimation(QString);

	void readAnimation(QString);

private slots:

//	void showHorizonWidget();

	//void computeCache();

	void add();
	void remove();

	void moveUp();
	void moveDown();
	void moveTop();
	void moveBottom();

	//	void addView3D();

//	void filterChanged();
	void trt_basketListSelectionChanged();
	void saveAnimation(bool actif);
	void playAnimation(bool );
	void moveAnimation(int);
	void updateAnimation();

	void speedChanged(int);

	void dataRemoved(IData* );
	void dataAdded(IData*);
	void orderChangedFromData(int oldIndex, int newIndex);

	//void view3DChanged(int i);
	void attributChanged(int i);


	void setRangeFromImage(unsigned int index,QVector2D range);
	void setRangeToImage(QVector2D range);


	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);

	void setRangeLock(unsigned int ,const QVector2D &range);


private:
	void fillComboAttribut();
	void warnBadAttribut(const QString& attributName);

	HorizonDataRep *m_rep;

	RGBPaletteWidget* m_paletteRGB;
	PaletteWidget* m_palette;
	QPointer<CUDARGBInterleavedImage> m_lastimage = nullptr;

	bool m_modeRGB = true;
	bool m_lastRGB = true;

	QListWidget* m_orderListWidget;

	QCheckBox* m_lockPalette;
	QComboBox* m_comboAttribut;
	//QComboBox* m_comboView3D;
	QToolButton* m_playButton;
	QToolButton* m_saveButton;
	QSlider* m_animSlider;
	QSlider* m_speedSlider;

	QTimer* m_animTimer;
	int m_speedAnim = 500;
	int m_lastSelected = -1;

	int m_lastValue=-1;


	bool m_cacheGPU = false;
	int m_cacheIndex = 0;

	QMutex m_orderMutex;
};

#endif
