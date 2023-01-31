#ifndef SAVELAYERDIALOG_H
#define SAVELAYERDIALOG_H

#include <QDialog>

class QListWidgetItem;
class QListWidget;
class QLineEdit;
class QCheckBox;

class SaveHorizonDialog : public QDialog {
Q_OBJECT
public:
	SaveHorizonDialog(const QStringList& saveNames, QString title, QWidget *parent = nullptr,
			Qt::WindowFlags f = Qt::WindowFlags());
	~SaveHorizonDialog();

	QString getSaveName();
	bool doInterpolation();
	bool doRgb();
	bool isNameNew();

private slots:
	void slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous);
	void updateInterpolationToggle(int state);
	void updateRgbToggle(int state);
	void setNewLayerName();

private:
	const QStringList& m_saveNames;
	QString m_outputSaveName = "";
	bool m_doInterpolation = true;
	bool m_doRgb = true;
	bool m_isNameNew = false;

	QListWidget* m_listWidget;
	QLineEdit* m_newNameLineEdit;
	QCheckBox* m_interpolationCheckBox;
	QCheckBox* m_rgbCheckBox;

};

#endif
