#ifndef FixedLayerFromDatasetPropPanel_H
#define FixedLayerFromDatasetPropPanel_H

#include <QWidget>

class QLineEdit;
class FixedLayerFromDatasetRep;
class PaletteWidget;
class QComboBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class FixedLayerFromDatasetPropPanel: public QWidget {
	Q_OBJECT
public:
	FixedLayerFromDatasetPropPanel(FixedLayerFromDatasetRep *rep, bool for3D,QWidget *parent);
	virtual ~FixedLayerFromDatasetPropPanel();
	void updatePalette();
	void updateComboValue(QString value);
private slots:
	void layerChanged(QString val);
	void showCrossHair(int value);

	void pick();
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

protected:
	QWidget* createComboBox();

	void addNewItemToComboBoxList(QString item);
private:
	QLineEdit *m_window;
	FixedLayerFromDatasetRep *m_rep;
	QComboBox* m_layerComboBox;
	QGroupBox *m_layerBox;
	PaletteWidget *m_palette;

	QCheckBox * m_showCrossHair;

	QToolButton * m_pickButton;
	PointPickingTask * m_pickingTask;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
