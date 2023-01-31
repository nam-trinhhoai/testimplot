#ifndef COLORTABLECHOOSER_H_
#define COLORTABLECHOOSER_H_

#include <QDialog>

class ColorTable;
class ColorTableSelector;

class QComboBox;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTableWidgetItem;

class ColorTableChooser : public QDialog {
	Q_OBJECT
public:
	ColorTableChooser(ColorTableSelector* from, QWidget* parent=0);
	virtual ~ColorTableChooser();

	void selectLineFromData(const ColorTable&);

private slots:
	void applyFilter();
	void filterChanged(int);
	void ciChange();

signals:
	void selectionChanged(const ColorTable&);

private:
	bool filterColorTable(const ColorTable& col, const std::string& family);
	int getNTotalRows();
	int populateTableWidget();

	ColorTableSelector* m_from;
	QTableWidget* m_table;
	QLineEdit* m_filterData;
	QComboBox* m_filterType;
	QPushButton* m_filterApply;

	int m_idxFilter;
	int m_curRow=-1;
	int m_curCol=-1;
};

#endif /* COLORTABLECHOOSER_H_ */
