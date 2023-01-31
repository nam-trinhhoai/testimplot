#ifndef SaveGraphicLayerDialog_H
#define SaveGraphicLayerDialog_H

#include <QDialog>

class QLineEdit;

class SaveGraphicLayerDialog: public QDialog {
Q_OBJECT
public:
	SaveGraphicLayerDialog(QWidget *parent);
	virtual ~SaveGraphicLayerDialog();

	QString fileName();

protected:
	QLineEdit *m_GraphicLayerFileLineEdit = nullptr;
};

#endif
