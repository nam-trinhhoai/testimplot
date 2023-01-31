#ifndef SRC_WIDGET_DATAMANAGER_FILEINFORMATIONTABLEWIDGET_H_
#define SRC_WIDGET_DATAMANAGER_FILEINFORMATIONTABLEWIDGET_H_

#include <QWidget>
#include <QFileInfo>

class QTableWidget;
class MonoFileBasedData;

class FileInformationTableWidget : public QWidget {
	Q_OBJECT
public:
	FileInformationTableWidget(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~FileInformationTableWidget();
	void clear();
	void addData(const QList<MonoFileBasedData>& newData);
private:
	QList<MonoFileBasedData> m_data;
	QTableWidget* m_tableWidget;
};

class MonoFileBasedData {
public:
	MonoFileBasedData(const QString& name, const QString& path);
	~MonoFileBasedData();

	QString name() const;
	QString path() const;
	bool isValid() const;
	QString owner() const;
	QDateTime birthDate() const;
private:
	QString m_name;
	QString m_path;
	QFileInfo m_fileInfo;

};

#endif
