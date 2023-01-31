#ifndef FixedRGBLayersFromDatasetPropPanel_H
#define FixedRGBLayersFromDatasetPropPanel_H

#include <QWidget>

class FixedRGBLayersFromDatasetRep;
class RGBPaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;

class FixedRGBLayersFromDatasetPropPanel : public QWidget{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetPropPanel(FixedRGBLayersFromDatasetRep *rep,QWidget *parent);
	virtual ~FixedRGBLayersFromDatasetPropPanel();

	void updatePalette(int i);

private:
	void changeDataKeyFromSlider(long index);
	QWidget* createImageChooserWidget();

	FixedRGBLayersFromDatasetRep *m_rep;
	RGBPaletteWidget * m_palette;
};

#endif
