
#ifndef __DATAFILECREATIONINFO__
#define __DATAFILECREATIONINFO__

#include <vector>
#include <string>

class QString;

class DataFileCreationInfo
{
public:
	DataFileCreationInfo();
	DataFileCreationInfo(QString path);
	virtual ~DataFileCreationInfo();
	void clear();
	void addComment(QString type, int val);
	void addComment(QString type, QString val);
	void addComment(QString type, double val);
	void addComment(QString type, long val);
	void addComment(QString type, std::string val);
	void setDataPath(QString path);
	void setDataName(QString name);
	std::string write();

private:
	QString header();
	QString getUserName();
	QString getMessagePath();
	QString currentTime();
	QString currentDate();
	QString getFilename();
	QString getDirectoryPath();
	QString getCompleteBaseName();

	std::vector<QString> m_message;
	QString fileHeader = "";
	QString m_dataPath = "";
	QString m_dataName = "";
};

#endif
