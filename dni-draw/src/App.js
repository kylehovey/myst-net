import React, { Component, Fragment } from 'react';
import Draw from './draw';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className ="app">
        <Fragment>
          <div className="app-header">
            <h1>D'ni Numeral Reader</h1>
          </div>
          <Draw />
        </Fragment>
      </div>
    );
  }
}

export default App;
