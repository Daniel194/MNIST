import { MnistPage } from './app.po';

describe('mnist App', () => {
  let page: MnistPage;

  beforeEach(() => {
    page = new MnistPage();
  });

  it('should display welcome message', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('Welcome to app!');
  });
});
