#include "datamanager/util_filesystem.h"

#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QDebug>
#include <QMessageBox>
#include <QInputDialog>
#include <QProcess>
#include <QLocale>

// return true if success with QStringList being folders
// folders are ordered as this : i+1 folder is parent of i folder
QPair<bool, QStringList> mkpath(QString path) {
	path = QDir::cleanPath(path); // following code will suppose that there are not . .. and "//" and empty strings
	bool success = QFileInfo(path).exists();
	QStringList newDirs;
	QString currentPath = path;
	QString lastPath = "";
	while (!QFileInfo(currentPath).exists() && lastPath.compare(currentPath)!=0) { // to avoid infinite loops
		QFileInfo fileInfo(currentPath);
		lastPath = currentPath;
		currentPath = fileInfo.dir().absolutePath();
		newDirs << fileInfo.absoluteFilePath();
	}
	QFileInfo workingFileInfo(currentPath);
	success = workingFileInfo.exists() && workingFileInfo.isDir();
	if (success && newDirs.count()>0) {
		long i = newDirs.count()-1;
		while (i>=0 && success) {
			QFileInfo toCreateDir(newDirs[i]);
			QDir parentDir = toCreateDir.dir(); // should exists
			success = parentDir.mkdir(toCreateDir.fileName());
			if (success) {
				i--;
			}
		}
		if (!success) {// fail to create dirs so need to delete what was created
			// i failed so i+1 was last created
			for (long removeIndex=i+1; removeIndex<newDirs.count(); removeIndex++) {
				QFileInfo toDeleteDir(newDirs[removeIndex]);
				QDir parentDir = toDeleteDir.dir(); // should exists
				bool deleteSuccess = parentDir.rmdir(toDeleteDir.fileName());
				if (!deleteSuccess) {
					qDebug() << "ERROR : Failed to delete just created directory " << toDeleteDir.absoluteFilePath();
				}
			}
		}
	}

	return QPair<bool, QStringList>(success, newDirs);
}

// remove from first index to last index
bool rmpath(QStringList paths) {
	bool success = true;
	long i=0;
	while (i<paths.count() && success) {
		QFileInfo toDeleteDir(paths[i]);
		QDir parentDir = toDeleteDir.dir(); // should exists
		success = parentDir.rmdir(toDeleteDir.fileName());
		if (!success) {
			qDebug() << "ERROR : Failed to delete directory " << toDeleteDir.absoluteFilePath();
		} else {
			i++;
		}
	}
	if (!success) {
		// recreate deleted dirs
		for (long createIndex=i-1; createIndex>=0; createIndex--) {
			QFileInfo toCreateDir(paths[createIndex]);
			QDir parentDir = toCreateDir.dir(); // should exists
			bool createSuccess = parentDir.mkdir(toCreateDir.fileName());
			if (!createSuccess) {
				qDebug() << "ERROR : Failed to create just deleted directory " << toCreateDir.absoluteFilePath();
			}
		}
	}
	return success;
}

bool stringListMatch(const QStringList& smallList, const QStringList& longList, long startIndex) {
	bool out = startIndex + smallList.count() <= longList.count();
	if (out) {
		long i = 0;
		while (out && i<smallList.count()) {
			out = smallList[i].compare(longList[i+startIndex])==0;
			i++;
		}
	}
	return out;
}

QString moveFileBetweenDirSystems(QString oriPath, const QStringList& _oriDirSystem, const QStringList& _targetDirSystem) {
	QString trashPathOut;
	oriPath = QDir::cleanPath(oriPath); // following code will suppose that there are not . .. and "//" and empty strings

	QStringList oriDirSystem, targetDirSystem;
	oriDirSystem << "ImportExport" << _oriDirSystem;


	// detect ImportExport/IJK
	QStringList breakDownPath = oriPath.split("/");
	QStringList cumulPath;
	long index = 0;
	while (index<breakDownPath.count()-1 && !stringListMatch(oriDirSystem, breakDownPath, index)) {
		cumulPath << breakDownPath[index];
		index++;
	}
	if (index<breakDownPath.count()-1) {
		cumulPath << breakDownPath[index]; // to add ImportExport
		QString importExportDirPath = cumulPath.join("/");


		// contruct the path with mirroring of initial structure
		QStringList remainingPathList;
		remainingPathList << _targetDirSystem;
		for (long remainIndex=index+oriDirSystem.count(); remainIndex<breakDownPath.count()-1; remainIndex++) {
			remainingPathList << breakDownPath[remainIndex];
		}
		QString endName =  breakDownPath[breakDownPath.count()-1];

		QDir movingDir(importExportDirPath);
		QString remainingPath = remainingPathList.join("/");
		bool mkPathValid = movingDir.mkpath(remainingPath);
		bool renameValid = false;


		if (mkPathValid) {
			trashPathOut = importExportDirPath + "/" + remainingPathList.join("/") + "/" + endName;
			renameValid = QFile(oriPath).rename(trashPathOut);
		}
		if (mkPathValid && !renameValid) {
			QDir cleanUpDir(importExportDirPath + "/" + remainingPathList.join("/"));
			trashPathOut = "";
		} else if (mkPathValid) {
			// try clean up old path
			bool goOn = true;
			QDir cleanUpDir(QFileInfo(oriPath).dir());
			while (goOn) {
				QString oldName = cleanUpDir.dirName();
				goOn = cleanUpDir.cdUp();
				goOn = goOn && cleanUpDir.rmdir(oldName);
			}
		}
	}

	return trashPathOut;
}

QString moveFileToTrash(QString oriPath) {
	return moveFileBetweenDirSystems(oriPath, QStringList() << "IJK", QStringList() << "IJK_Trash" << "IJK");
}

QString restoreFileFromTrash(QString oriPath) {
	return moveFileBetweenDirSystems(oriPath, QStringList() << "IJK_Trash" << "IJK", QStringList() << "IJK");
}

std::pair<bool, DeletableLeaf> ijkMoveLeaf(DeletableLeaf leaf, bool toTrash) {
	const QStringList& paths = leaf.paths();
	long index = paths.count()-1;
	bool goOn = true;
	QList<QPair<QString, QString>> errorRestoreSafetyList;
	while(index>=0 && goOn) {
		QString trashPath;
		if (toTrash) {
			trashPath = moveFileToTrash(paths[index]); // if function fail we expect it to restore the failing file
		} else {
			trashPath = restoreFileFromTrash(paths[index]);
		}
		goOn = !trashPath.isNull() && !trashPath.isEmpty();
		if (goOn) {
			errorRestoreSafetyList.append(QPair<QString, QString>(paths[index], trashPath));
			index--;
		}
	}
	std::pair<bool, DeletableLeaf> result;
	result.first = index<0;
	if (!result.first) {
		QString trashTitle, trashErrorMessage, trashRestoreMessage;
		if (toTrash) {
			trashTitle = "File Trashing failure";
			trashErrorMessage = "Failed to trash file ";
			trashRestoreMessage = "Failed to restore ";
		} else {
			trashTitle = "File Restoration failure";
			trashErrorMessage = "Failed to restore file ";
			trashRestoreMessage = "Failed to return file to trash ";
		}
		// restore in opposite order
		bool restoreWork;
		QMessageBox::warning(nullptr, trashTitle, trashErrorMessage + paths[index]);
		for (long restoreIndex=index+1; restoreIndex<paths.count(); restoreIndex++) {
			QString restoredFile = restoreFileFromTrash(errorRestoreSafetyList[paths.count()-restoreIndex].second);
			restoreWork = !restoredFile.isNull() && !restoredFile.isEmpty() && restoredFile.compare(errorRestoreSafetyList[paths.count()-restoreIndex].first);
			if (!restoreWork) {
				QMessageBox::warning(nullptr, trashTitle, trashRestoreMessage + errorRestoreSafetyList[paths.count()-restoreIndex].first);
			}
		}
	} else {
		QStringList trashPaths;
		for (const QPair<QString, QString>& pair : errorRestoreSafetyList) {
			trashPaths << pair.second;
		}
		result.second = DeletableLeaf(leaf.name(), trashPaths, leaf.parentName(), !leaf.isLoneChild());
	}
	return result;
}

std::pair<bool, DeletableLeaf> ijkMoveToTrash(DeletableLeaf leaf) {
	return ijkMoveLeaf(leaf, true);
}

std::pair<bool, DeletableLeaf> ijkRestoreFromTrash(DeletableLeaf leaf) {
	return ijkMoveLeaf(leaf, false);
}

bool recursiveCheckValid(QString path) {
	QFileInfo fileInfo(path);
	bool isValid = fileInfo.exists() && fileInfo.isWritable();
	if (isValid && fileInfo.isDir()) {
		QDir dir(path);
		QFileInfoList childInfoLists = dir.entryInfoList(QStringList() << "*", QDir::NoDotAndDotDot |
						QDir::Dirs | QDir::Files | QDir::System | QDir::Hidden);
		std::size_t idx = 0;
		while (isValid && idx<childInfoLists.count()) {
			isValid = recursiveCheckValid(childInfoLists[idx].absoluteFilePath());
			idx++;
		}
	}
	return isValid;
}

std::pair<bool, DeletableLeaf> ijkDeleteFromTrash(DeletableLeaf leaf, QString logPath) {
	DeletableLeaf reducedLeaf = leaf; // only used if failure during suppression

	// ask for consent
	QStringList paths = leaf.paths();
	int fileNumber = paths.count();

	QString filesStr = paths.join("\n");

	QString deleteStr = "Delete";
	QString abortStr = "Abort";

	QStringList items;
	items << abortStr << deleteStr;

	QString selectedItem = QInputDialog::getItem(nullptr, "Confirm Deletion", "You are about to delete "+QString::number(fileNumber)+" items.\n"+filesStr+"\nDo you wish to proceed", items);
	bool allowDelete = selectedItem.compare(deleteStr)==0;

	bool canBeDeleted = false;
	if (allowDelete) {
		bool rightsValid = true;
		std::size_t idx = 0;
		while (idx<paths.count() && rightsValid) {
			QString path = paths[idx];
			rightsValid = recursiveCheckValid(path); //QFileInfo(path).exists() && QFileInfo(path).isWritable();
			idx++;
		}
		canBeDeleted = rightsValid;

		if (!canBeDeleted) {
			QMessageBox::information(nullptr, "Permissions error", "Can not delete items because permissions do not allow deletion");
		}
	}

	bool deletionDone = false;
	if (allowDelete && canBeDeleted) {
		std::size_t idxDelete = 0;
		bool result = true;
		while (result && idxDelete<paths.count()) {
			QString path = paths[idxDelete];
			result = deletePath(path, logPath);
			if (!result) {
				// ask to contact support to fix impossible case issue
				// stop suppression
				QMessageBox::critical(nullptr, "Unforeseen deletion failure", "Deletion failed while checks did not find any issue. Data may become inconsistent. Please contact support.");
			} else {
				idxDelete++;
			}
		}
		if (result) {
			for (QString path : paths) {
				// try clean up old path
				bool goOn = true;
				QDir cleanUpDir(QFileInfo(path).dir());
				while (goOn) {
					QString oldName = cleanUpDir.dirName();
					goOn = cleanUpDir.cdUp();
					goOn = goOn && cleanUpDir.rmdir(oldName);
				}
			}
			deletionDone = true;
		} else {
			QStringList trashPaths;
			for (std::size_t idxRemaining=idxDelete; idxRemaining<paths.count(); idxRemaining++) {
				trashPaths << paths[idxRemaining];
			}
			reducedLeaf = DeletableLeaf(leaf.name(), trashPaths, leaf.parentName(), !leaf.isLoneChild());
		}
	}

	return std::pair<bool, DeletableLeaf>(deletionDone, reducedLeaf);
}

bool deleteFile(QString path, QString logPath) {
	QProcess process;
	process.start("whoami");
	process.waitForFinished();

	QString deletor;
	if (process.exitCode()==QProcess::NormalExit) {
		deletor = process.readAllStandardOutput();
	} else {
		deletor = "UserNotFound";
	}

	QFileInfo fileInfo(path);
	QLocale locale;
	QString logStr = "Deletor : " + deletor + ", Owner : " + fileInfo.owner() +
			", Creation Date : " +  locale.toString(fileInfo.birthTime(),  "dd.MM.yyyy hh:mm") +
			", Last Modification : " + locale.toString(fileInfo.lastModified(),  "dd.MM.yyyy hh:mm") + ", File : " + path;
	bool out = QFile(path).remove();

	if (out) {
		logStr = "Deletion success : " + logStr;
	} else {
		logStr = "Deletion failure : " + logStr;
	}
	addToLog(logPath, logStr);
	qDebug() << logStr;
	return out;
}

bool deleteDir(QString path, QString logPath) {
	bool valid = true;
	bool childRemaining = true;
	QDir dir(path);
	while (valid && childRemaining) {
		QFileInfoList childInfoLists = dir.entryInfoList(QStringList() << "*", QDir::NoDotAndDotDot |
				QDir::Dirs | QDir::Files | QDir::System | QDir::Hidden);
		if (childInfoLists.count()>0) {
			QString childPath = childInfoLists.first().absoluteFilePath();
			valid = deletePath(childPath, logPath);
			if (childInfoLists.count()<=1) {
				childRemaining = false;
			}
		}
	}
	if (valid) {
		QString oldName = dir.dirName();
		valid = dir.cdUp();
		if (valid) {
			QProcess process;
			process.start("whoami");
			process.waitForFinished();

			QString deletor;
			if (process.exitCode()==QProcess::NormalExit) {
				deletor = process.readAllStandardOutput();
			} else {
				deletor = "UserNotFound";
			}

			QFileInfo fileInfo(path);
			QLocale locale;
			QString logStr = "Deletor : " + deletor + ", Owner : " + fileInfo.owner() +
					", Creation Date : " +  locale.toString(fileInfo.birthTime(),  "dd.MM.yyyy hh:mm") +
					", Last Modification : " + locale.toString(fileInfo.lastModified(),  "dd.MM.yyyy hh:mm") + ", Dir : " + path;

			valid = dir.rmdir(oldName);

			if (valid) {
				logStr = "Deletion success : " + logStr;
			} else {
				logStr = "Deletion failure : " + logStr;
			}
			addToLog(logPath, logStr);
			qDebug() << logStr;
		} else {
			qDebug() << "Deletion rejected because QDir::cdUp return false";
		}
	} else {
		qDebug() << "Deletion rejected all files could not be deleted";
	}
	return valid;
}

bool deletePath(QString path, QString logPath) {
	bool out = false;
	if (QFileInfo(path).exists() && QFileInfo(path).isWritable()) {
		if (QFileInfo(path).isDir()) {
			out = deleteDir(path, logPath);
		} else {
			out = deleteFile(path, logPath);
		}
	} else {
		qDebug() << "Deletion rejected because of rights";
	}
	return out;
}

void addToLog(QString logPath, QString logText) {
	QFile file(logPath);
	if (file.open(QFile::Append | QFile::Text)) {
		QTextStream stream(&file);
		stream << logText << "\n";
	} else {
		qDebug() << "Failed to log message";
	}
}
