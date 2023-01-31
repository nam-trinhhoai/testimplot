#include <QDialog>

class QListWidget;

class WorkingSetManager;
class WellBore;

class SelectRandomCreationMode : public QDialog {
	Q_OBJECT
public:
	SelectRandomCreationMode(WorkingSetManager* manager, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~SelectRandomCreationMode();

	QList<WellBore*> selectedWellBores() const;
	double wellMargin() const;

private slots:
	void itemSelectionChanged();
	void setValueMargin(double);

private:
	QListWidget* m_listWidget;

	QList<WellBore*> m_selectedWellBores;
	QList<WellBore*> m_wellBores;
	double m_wellMargin;
};
