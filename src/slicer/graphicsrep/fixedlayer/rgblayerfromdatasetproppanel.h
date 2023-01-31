#ifndef RgbLayerFromDatasetPropPanel_H
#define RgbLayerFromDatasetPropPanel_H

#include <QWidget>

class QLineEdit;
class RgbLayerFromDatasetRep;
class PaletteWidget;
class QComboBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class RgbLayerFromDatasetPropPanel: public QWidget {
	Q_OBJECT
public:
	RgbLayerFromDatasetPropPanel(RgbLayerFromDatasetRep *rep, bool for3D,QWidget *parent);
	virtual ~RgbLayerFromDatasetPropPanel();
	void updatePalette();
	void updateComboValueRed(QString value);
	void updateComboValueGreen(QString value);
	void updateComboValueBlue(QString value);
private slots:
	void layerRedChanged(QString val);
	void layerGreenChanged(QString val);
	void layerBlueChanged(QString val);

	void pick();
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

protected:
	QWidget* createComboBox();

	void addNewItemToComboBoxList(QString item);
private:
	QLineEdit *m_window;
	RgbLayerFromDatasetRep *m_rep;
	QComboBox* m_layerComboBoxRed;
	QComboBox* m_layerComboBoxGreen;
	QComboBox* m_layerComboBoxBlue;
	QGroupBox *m_layerBox;
	PaletteWidget *m_paletteRed;
	PaletteWidget *m_paletteGreen;
	PaletteWidget *m_paletteBlue;

	QToolButton * m_pickButton;
	PointPickingTask * m_pickingTask;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
