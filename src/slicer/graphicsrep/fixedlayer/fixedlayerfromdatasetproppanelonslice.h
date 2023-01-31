#ifndef FixedLayerFromDatasetPropPanelOnSlice_H
#define FixedLayerFromDatasetPropPanelOnSlice_H

#include <QWidget>
#include "pickinginfo.h"

class FixedLayerFromDatasetRepOnSlice;
class QCheckBox;
class AbstractView;
class QPushButton;

class FixedLayerFromDatasetPropPanelOnSlice: public QWidget {
	Q_OBJECT
public:
	FixedLayerFromDatasetPropPanelOnSlice(FixedLayerFromDatasetRepOnSlice *rep, QWidget *parent);
	virtual ~FixedLayerFromDatasetPropPanelOnSlice();

	void setColor(QColor);
private:
	void selectColor();

	FixedLayerFromDatasetRepOnSlice *m_rep;

	QPushButton* m_colorButton;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
