#ifndef StratiSliceChooserDialog_H
#define StratiSliceChooserDialog_H

#include <QDialog>

class QComboBox;
class SeismicSurvey;
class Seismic3DAbstractDataset;
class StratiSliceChooserDialog: public QDialog {
Q_OBJECT
public:
	StratiSliceChooserDialog(SeismicSurvey *survey, QWidget *parent);
	virtual ~StratiSliceChooserDialog();

	Seismic3DAbstractDataset *seismic();
	Seismic3DAbstractDataset *rgt();

protected:
	QComboBox *m_seismicCombo = nullptr;
	QComboBox *m_rgtCombo = nullptr;
};

#endif
