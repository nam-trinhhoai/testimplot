#ifndef SRC_SLICER_DIALOG_BNNICONFIGCREATIONDIALOG_H
#define SRC_SLICER_DIALOG_BNNICONFIGCREATIONDIALOG_H

#include <QDialog>
#include <QStringList>

class QDialogButtonBox;
class QCheckBox;
class QComboBox;
class QLineEdit;

class BnniConfigCreationDialog : public QDialog {
	Q_OBJECT
public:
	BnniConfigCreationDialog(const QStringList& existingConfigs, const QStringList& checkpoints,
			QWidget* parent=nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~BnniConfigCreationDialog();

	bool useCheckpoints() const;
	void toggleUseCheckpoints(bool);

	QString checkpoint() const;
	bool checkpointValid() const;
	QString newName() const;

private slots:
	void checkpointChanged(int val);
	void nameChanged();
	void useCheckpointsChanged(int val);

private:
	QStringList m_checkpoints;
	QStringList m_existingConfigs;
	bool m_useCheckpoints = false;
	QString m_newName;
	int m_currentCheckpoint = -1;

	QDialogButtonBox* m_buttonBox;
	QComboBox* m_checkpointsComboBox;
	QLineEdit* m_nameLineEdit;
	QCheckBox* m_useCheckpointsCheckBox;
};

#endif
