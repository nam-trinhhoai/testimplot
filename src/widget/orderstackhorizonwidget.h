#ifndef ORDERSTACKHORIZONWIDGET_H
#define ORDERSTACKHORIZONWIDGET_H

#include "workingsetmanager.h"
#include "geotimegraphicsview.h"
#include <QWidget>
#include <memory>


class FixedRGBLayersFromDatasetAndCube;

class QLineEdit;
class QListWidget;
class QPushButton;
class QSlider;
class QToolButton;
class QTimer;

class OrderStackHorizonWidget : public QWidget {
	Q_OBJECT
public:

	OrderStackHorizonWidget(GeotimeGraphicsView* graphicsview, WorkingSetManager* workingset, QWidget* parent=nullptr);
	~OrderStackHorizonWidget();


	QStringList getAttributesAvailable();


public slots:

	void add();
	void remove();

	void moveUp();
	void moveDown();
	void moveTop();
	void moveBottom();

	void apply();
	void addView3D();

//	void filterChanged();
	void trt_basketListSelectionChanged();
	void playAnimation(bool );
	void moveAnimation(int);
	void updateAnimation();

	void speedChanged(int);

	void dataRemoved(IData* );
	void dataAdded(IData*);

	void view3DChanged(int i);
	void attributChanged(int i);



private:
	WorkingSetManager* m_workingset;
	GeotimeGraphicsView* m_graphicsview;

	//QListWidget* m_listItemToSelectWidget;
	QListWidget* m_orderListWidget;

	QList<int> m_indexList;

	QComboBox* m_comboAttribut;
	QComboBox* m_comboView3D;
	QToolButton* m_playButton;
	QSlider* m_animSlider;
	QSlider* m_speedSlider;

	QTimer* m_animTimer;
	int m_indexAnim;
	int m_index3D = 1;

	QList<int>  m_viewIndex;
	QList<QPair<int,int>> m_viewAttribIndex;

	int m_lastSelected = -1;

	int m_speedAnim = 500;
	//QLineEdit* m_filterEdit
	QList<FixedRGBLayersFromDatasetAndCube*> m_dataOrderList;




};

#endif // ORDERSTACKHORIZONWIDGET_H
