#ifndef FixedRGBLayersFromDatasetPropPanelOnSlice_H
#define FixedRGBLayersFromDatasetPropPanelOnSlice_H

#include <QWidget>

class FixedRGBLayersFromDatasetRepOnSlice;
class QSlider;
class QSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;

class FixedRGBLayersFromDatasetPropPanelOnSlice : public QWidget{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetPropPanelOnSlice(FixedRGBLayersFromDatasetRepOnSlice *rep,QWidget *parent);
	virtual ~FixedRGBLayersFromDatasetPropPanelOnSlice();

private:
	void changeDataKeyFromSlider(long index);

	FixedRGBLayersFromDatasetRepOnSlice *m_rep;
};

#endif
